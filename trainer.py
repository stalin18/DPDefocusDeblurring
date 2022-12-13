import os
import time
import numpy as np

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

from utils.logger import Logger
from utils.utils import AverageMeter
from utils.utils import CharbonnierLoss
from pytorch_msssim import MS_SSIM


class Trainer:
    def __init__(self, args, rank, world_size):
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        self.args = args
        self.rank = rank
        self.world_size = world_size

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

        # load and finetune from checkpoint if available
        if self.args.checkpoint is not None:
            print("Loading checkpoint: '{}'".format(self.args.checkpoint))
            checkpoint = torch.load(self.args.checkpoint)

            pretrained_dict = checkpoint
            model_dict = self.net.state_dict()

            # models are trained in DDP, remove module prefix for proper loading
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict.keys()}

            self.net.load_state_dict(pretrained_dict, strict=True)

            print('Loaded checkpoint model!')

        # Synchronize BatchNorm layers before replicating on multiple GPUs
        self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net.to(rank, non_blocking=True)
        self.net = DDP(self.net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        """ define train and validation dataloaders """
        train_dataset = DPDefocusDataset(data_dir=args.data_dir, mode='train')
        # create DistributedSampler, shuffle the data within the sampler so that all GPUs receive the same shuffled data
        sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True,
                                     drop_last=False)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                           shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True, drop_last=False,
                                           sampler=sampler)

        # create the val dataset
        val_dataset = DPDefocusDataset(data_dir=args.data_dir, mode='val')
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                         shuffle=True, num_workers=args.num_workers,
                                         pin_memory=True, drop_last=False,
                                         sampler=None)

        self.train_dataloader_iter = iter(self.train_dataloader)
        self.iters_per_epoch = len(self.train_dataloader)

        print('Dataset size: %d' % len(self.train_dataloader.dataset)) if rank == 0 else None
        print('Total number of training epochs: %d' % self.args.num_epochs) if rank == 0 else None
        print('Dataset size: %d' % self.iters_per_epoch) if rank == 0 else None
        print('Dataset size: %d' % (self.args.num_epochs * self.iters_per_epoch)) if rank == 0 else None

        """ get the logger path and setup SummaryWriter """
        if self.rank == 0:
            logger_path = self.get_logger_path(self.args)
            self.logger = Logger(logger_path)

        """ setup the criterion """
        if self.args.loss_type == None:
            self.mse_loss = nn.MSELoss()
            self.mse_loss = self.mse_loss.to(rank)
        elif self.args.loss_type == 'charbonnier':
            self.charbonnier_loss = CharbonnierLoss()
            self.charbonnier_loss = self.charbonnier_loss.to(rank)
        elif self.args.loss_type == 'mix':
            self.charbonnier_loss = CharbonnierLoss()
            self.msssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)

            self.charbonnier_loss = self.charbonnier_loss.to(rank)
            self.msssim = self.msssim.to(rank)

        """ setup the optimizer """
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, eps=1e-7)
        self.step_scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                        step_size=self.args.lr_decay_steps * self.iters_per_epoch,
                                                        gamma=self.args.lr_decay_factor)

    def run_training(self):
        start_time = time.time()

        for self.e in range(self.args.num_epochs):
            # set the model to train() mode
            self.net.train()

            # epoch average
            examples_per_sec = AverageMeter()
            data_time = AverageMeter()

            step_counter = 0
            tic = time.time()

            for i in range(0, self.iters_per_epoch):
                try:
                    batch_data = next(self.train_dataloader_iter)
                except StopIteration:
                    self.train_dataloader_iter = iter(self.train_dataloader)
                    batch_data = next(self.train_dataloader_iter)

                step_counter += 1
                # current iteration number
                current_iter = self.e * self.iters_per_epoch + step_counter

                data_time.update(time.time() - tic)

                # batch data
                (img_l, img_r, img_gt) = batch_data
                img_l = img_l.to(self.rank)
                img_r = img_r.to(self.rank)
                img_gt = img_gt.to(self.rank)

                # forward pass
                output = self.net(img_l, img_r)

                # loss computation
                if self.args.loss_type == None:
                    loss = self.args.loss_wt1 * self.mse_loss(output, img_gt)
                    loss_log = {'MSE Loss': loss}
                if self.args.loss_type == 'charbonnier':
                    loss = self.args.loss_wt1 * self.charbonnier_loss(output, img_gt)
                    loss_log = {'Charbonnier Loss': loss}
                if self.args.loss_type == 'mix':
                    charbonnier_loss = self.args.loss_wt1 * self.charbonnier_loss(output, img_gt)
                    msssim_loss = self.args.loss_wt2 * (1.0 - self.msssim(output, img_gt))
                    loss = charbonnier_loss + msssim_loss
                    loss_log = {'Charbonnier Loss': charbonnier_loss, 'MSSSIM Loss': msssim_loss}

                # backward pass
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.step_scheduler.step()

                # logging loss every step
                if self.rank == 0:
                    for tag, value in loss_log.items():
                        self.logger.scalar_summary(tag, value, current_iter)

                # logging learning rate, training inputs & outputs after vis_after steps
                if self.rank == 0 and current_iter != 0 and current_iter % self.args.vis_after == 0:
                    duration = time.time() - tic
                    examples_per_sec.update(self.args.batch_size / duration)

                    time_elapsed = (time.time() - start_time) / 3600
                    time_left = ((self.args.num_epochs * self.iters_per_epoch) / current_iter - 1.0) * time_elapsed
                    print('epoch: %d | e-iteration: %d/%d | total_iteration: %d | loss %f | examples/s %.2f | '
                          'time elapsed: %.2f h | time left: %.2f h' % (
                          self.e, i + 1, self.iters_per_epoch, current_iter, loss.data,
                          examples_per_sec.average(), time_elapsed, time_left)) if self.rank == 0 else None

                    self.logger.scalar_summary('learning rate', self.get_lr(self.optimizer), current_iter)

                    image_info = {'img_left': img_l,
                                  'img_right': img_r,
                                  'img_gt': img_gt,
                                  'output': output, }

                    for tag, value in image_info.items():
                        self.logger.image_summary(tag, value, current_iter, max_output=2)

                if self.args.check_validation:
                    break

                # start time for next batch
                tic = time.time()

            # save the model
            if self.rank == 0 and (self.e + 1) % self.args.save_after == 0:
                torch.save(self.net.state_dict(), os.path.join(self.args.checkpointdir, self.args.exp_name,
                                                               'model_e{}.pth'.format(self.e + 1)))

            # perform validation
            if self.rank == 0 and self.args.validate_after != 0 and (self.e + 1) % self.args.valudate_after == 0:
                self.validate()

            # other GPUs wait till the first GPU is done with loggin and validation
            dist.barrier()

    def validate(self):
        # validate
        print('Validating!')

        start_time = time.time()

        # set model to eval() mode
        self.net.eval()

        # loss and metrics init
        loss = 0
        psnr = 0
        ssim = 0
        mae = 0

        for i, batch_data in enumerate(self.val_dataloader):
            # avoid storing gradients
            with torch.no_grad():
                # batch data
                (img_l, img_r, img_gt) = batch_data
                img_l = img_l.to(self.rank)
                img_r = img_r.to(self.rank)
                img_gt = img_gt.to(self.rank)

                # forward pass
                output = self.net(img_l, img_r)

                # loss computation
                if self.args.loss_type == None:
                    loss = self.args.loss_wt1 * self.mse_loss(output, img_gt)
                if self.args.loss_type == 'charbonnier':
                    loss = self.args.loss_wt1 * self.charbonnier_loss(output, img_gt)
                if self.args.loss_type == 'mix':
                    charbonnier_loss = self.args.loss_wt1 * self.charbonnier_loss(output, img_gt)
                    msssim_loss = self.args.loss_wt2 * (1.0 - self.msssim(output, img_gt))

                # metrics computation
                img1 = np.transpose(img_gt[0].cpu().detach().numpy(), [1, 2, 0])
                img2 = np.transpose(output[0].cpu().detach().numpy(), [1, 2, 0])
                # argument order matters, img1 should be gt, img2 should be output
                temp_psnr, temp_ssim, temp_mae = self.measure_metrics(img1, img2)
                psnr += temp_psnr
                ssim += temp_ssim
                mae += temp_mae

                # save some validation image to tensorboard logger
                if i % 100 == 0:
                    image_info = {'val_img_left': img_l,
                                  'val_img_right': img_r,
                                  'val_img_gt': img_gt,
                                  'val_output': output, }

                    for tag, value in image_info.items():
                        self.logger.image_summary(tag, value, self.e + 1, max_output=2)

        if self.args.loss_type == 'mix':
            charbonnier_loss /= len(self.val_dataloader.dataset)
            msssim_loss /= len(self.val_dataloader.dataset)

        loss /= len(self.val_dataloader.dataset)
        psnr /= len(self.val_dataloader.dataset)
        ssim /= len(self.val_dataloader.dataset)
        mae /= len(self.val_dataloader.dataset)

        if self.args.loss_type == 'mix':
            print(f"Epoch: {self.e + 1} Loss: {loss} Charbonnier Loss: {charbonnier_loss} MSSSIM Loss: {msssim_loss} PSNR: {psnr}, SSIM: {ssim} MAE: {mae}")
        else:
            print(f"Epoch: {self.e + 1} Loss: {loss} PSNR: {psnr}, SSIM: {ssim} MAE: {mae}")

        average_loss_metrics = {'Loss': loss, 'PSNR': psnr, 'SSIM': ssim, 'MAE': mae}

        for tag, value in average_loss_metrics.items():
            self.logger.scalar_summary(tag, value, self.e + 1)


    def get_logger_path(self, args):
        if args.training_system == 'mlp':
            import mltracker
            logger_path = mltracker.get_tensorboard_dir()
        else:
            logger_path = os.path.join(args.checkpointdir, args.exp_name, 'logs')

        return logger_path

    def get_lr(self, optimizer):
        for p in optimizer.param_groups:
            current_lr = p['lr']
            print('Learning rate check: ', current_lr)

        return current_lr

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
