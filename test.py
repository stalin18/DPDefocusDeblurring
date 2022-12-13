import argparse
import os
import time
import numpy as np

import cv2

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from skimage import metrics
from sklearn.metrics import mean_absolute_error

from data.DPDefocusDataset import DPDefocusDataset

from model.dpdnet import DPDNet
from model.dpdd_panet1 import DPDD_PANet1
from model.dpdd_panet2 import DPDD_PANet2
from model.dpdd_panet2_1 import DPDD_PANet2_1
from model.dpdd_panet2_2 import DPDD_PANet2_2
from model.dpdd_panet2_pam_num1 import DPDD_PANet2_Num1
from model.dpdd_panet2_pam_num2 import DPDD_PANet2_Num2
from model.dpdd_panet2_pam_num3 import DPDD_PANet2_Num3
from model.dpdd_panet2_pam_num1_pos1 import DPDD_PANet2_Num1_Pos1
from model.dpdd_panet2_pam_num1_pos2 import DPDD_PANet2_Num1_Pos2
from model.dpdd_panet2_pam_num1_pos3 import DPDD_PANet2_Num1_Pos3


parser = argparse.ArgumentParser('Dual-Pixel based Defocus Deblurring')

parser.add_argument('--data_dir',
                    default='/kunal.swami/Workspaces/Datasets/DefocusDeblurning/dataset/DPDNet',
                    help= 'Dataset root directory')
parser .add_argument('--model', default='dpdnet', help= 'Model to use for testing, default is DPDNet')
parser.add_argument('--checkpointdir',
                    default='/kunal.swami/Workspaces/Datasets/DefocusDebLurcing/checkpoints',
                    help='Folder for model checkpoints and results')
parser .add_argument ('--model_name',
                      default=None,
                      heLp= 'Model folder name')
parser .add_argument ('--model_file',
                      default=None,
                      heLp= 'Model filename')


def measure_metrics(self, img1, img2):
    # define maximum pixel value
    PIXEL_MAX = 1.0

    # compute psnr
    psnr = metrics.peak_signal_noise_ratio(img1, img2)
    # compute ssim
    ssim = metrics.structural_similarity(img1, img2, data_range=PIXEL_MAX, channel_axis=2)
    # compute mae
    mae_0 = mean_absolute_error(img1[:, :, 0], img2[:, :, 0], multioutput='uniform_average')
    mae_1 = mean_absolute_error(img1[:, :, 1], img2[:, :, 1], multioutput='uniform_average')
    mae_2 = mean_absolute_error(img1[:, :, 2], img2[:, :, 2], multioutput='uniform_average')
    mae = np.mean([mae_0, mae_1, mae_2])

    return psnr, ssim, mae


class Tester:
    def __init__(self, args):
        self.args = args

        """ define and setup model """
        if args.model == 'dpdnet':
            self.net = DPDNet()
        if args.model == 'panet1':
            self.net = DPDD_PANet1()
        if args.model == 'panet2':
            self.net = DPDD_PANet2()
        if args.model == 'panet2_1':
            self.net = DPDD_PANet2_1()
        if args.model == 'panet2_2':
            self.net = DPDD_PANet2_2()
        if args.model == 'panet2_num1':
            self.net = DPDD_PANet2_Num1()
        if args.model == 'panet2_num2':
            self.net = DPDD_PANet2_Num2()
        if args.model == 'panet2_num3':
            self.net = DPDD_PANet2_Num3()
        if args.model == 'panet2_num1_pos1':
            self.net = DPDD_PANet2_Num1_Pos1()
        if args.model == 'panet2_num1_pos2':
            self.net = DPDD_PANet2_Num1_Pos2()
        if args.model == 'panet2_num1_pos3':
            self.net = DPDD_PANet2_Num1_Pos3()

        # load model checkpoint
        print("Loading checkpoint: '{}'".format(self.args.checkpoint))
        checkpoint = torch.load(self.args.checkpoint)

        pretrained_dict = checkpoint
        model_dict = self.net.state_dict()

        # models are trained in DDP, remove module prefix for proper loading
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                           if k.replace('module.', '') in model_dict.keys()}

        self.net.load_state_dict(pretrained_dict, strict=True)

        print('Loaded model for test!')

        self.net.cuda()

        """ define test dataloader """
        print("Entering the dataloader")

        # create the test dataset
        test_dataset = DPDefocusDataset(data_dir=args.data_dir, model='test')
        self.test_dataloader = DataLoader(test_dataset, batch_size=1,
                                          shuffle=False, num_workers=4,
                                          pin_memory=True, drop_last=False)

        self.test_dataloader_iter = iter(self.test_dataloader)

        print('Dataset size: %d' % len(self.test_dataloader.dataset))

        # create folder to save results
        save_path = os.path.join(self.args.checkpointdir, self.args.model_name, 'results')
        self.target_dir = os.path.join(save_path, self.args.model_file.split('.'[0]))
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def test(self):
        # test
        print('Testing!')

        # set model to eval() mode
        self.net.eval()

        # metrics init
        psnr = 0
        ssim = 0
        mae = 0

        for i, batch_data in enumerate(self.test_dataloader):
            # avoid storing gradients
            with torch.no_grad():
                # batch data
                (img_l, img_r, img_gt) = batch_data
                img_l = img_l.to(self.rank)
                img_r = img_r.to(self.rank)
                img_gt = img_gt.to(self.rank)

                # forward pass
                output = self.net(img_l, img_r)

                # get images
                gt = np.uint8(np.transpose(img_gt[0].cpu().detach().numpy(), [1, 2, 0]) * 255)
                output = np.uint8(np.transpose(output[0].cpu().detach().numpy(), [1, 2, 0]) * 255)
                dp_left = np.uint8(np.transpose(img_l[0].cpu().detach().numpy(), [1, 2, 0]) * 255)
                dp_right = np.uint8(np.transpose(img_r[0].cpu().detach().numpy(), [1, 2, 0]) * 255)

                gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                dp_left = cv2.cvtColor(dp_left, cv2.COLOR_RGB2BGR)
                dp_right = cv2.cvtColor(dp_right, cv2.COLOR_RGB2BGR)

                # metrics computation
                # argument matters, img1 should be gt, img2 should be output
                temp_psnr, temp_ssim, temp_mae = measure_metrics(np.float32(gt / 255.0), np.float32(output / 255.0))

                psnr += temp_psnr
                ssim += temp_ssim
                mae += temp_mae

                # save images with metrics in filename
                gt_name = str(i) + '_gt' + '.png'
                output_name = str(i) + '_' + str(round(psnr, 3)).replace('.', 'd') + '_' + \
                              str(round(ssim, 3)).replace('.', 'd') + '_' + str(round(mae, 3)).replace('.', 'd') + '.png'
                dp_left_name = str(i) + '_l' + '.png'
                dp_right_name = str(i) + '_r' + '.png'
                cv2.imwrite(os.path.join(self.target_dir, gt_name), gt)
                cv2.imwrite(os.path.join(self.target_dir, output_name), output)
                cv2.imwrite(os.path.join(self.target_dir, dp_left_name), dp_left)
                cv2.imwrite(os.path.join(self.target_dir, dp_right_name), dp_right)

                print(f"Tested : {i} PSNR: {temp_psnr} SSIM: {temp_ssim} MAE: {temp_mae}")

            psnr /= len(self.test_dataloader.dataset)
            ssim /= len(self.test_dataloader.dataset)
            mae /= len(self.test_dataloader.dataset)

            print(f"Test dataset size : {len(self.test_dataloader.dataset)} PSNR: {psnr} SSIM: {ssim} MAE: {mae}")


def main():
    args = parser.parse_args()

    tester = Tester(args)
    tester.test()

if __name__ == '__main__':
    main()