import os
from pathlib import Path

import cv2
from PIL import Image
import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils.utils import check_and_copy

def data_augmentations(img_l, img_r, img_gt):
    # randomly upscale patches between 512 & 640 (select even number for center crop later)
    up_size = int(np.random.randint(512, 640) / 2) * 2

    img_r = cv2.resize(img_r, (up_size, up_size), cv2.INTER_CUBIC)
    img_l = cv2.resize(img_l, (up_size, up_size), cv2.INTER_CUBIC)
    img_gt = cv2.resize(img_gt, (up_size, up_size), cv2.INTER_CUBIC)

    random_crop = int(np.random. randint(0, up_size - 511))
    img_r = img_r[random_crop: random_crop + 512, random_crop: random_crop + 512]
    img_l = img_l[random_crop:random_crop + 512, random_crop: random_crop + 512]
    img_gt = img_gt[random_crop:random_crop + 512, random_crop: random_crop + 512]
    # randomly flip patch horizontally, swap if flipped
    # flip_prob = np. random. random ()
    # if flip_prob < 0.5:
    #    img_r = cv2. flip (img_r, 0)
    #    img_l = cv2. flip (img_L, 0)
    #    img_gt = cv2. flip (img_gt, 0)

    return img_l, img_r, img_gt


transform = transforms.Compose([transforms.ToTensor()])


class DPDefocusDataset(Dataset):
    def _init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode  # train / val / test

        self.train_gt_list, self.train_l_list, self.train_r_list = [], [], []
        self.val_gt_list, self.val_l_list, self.val_r_list = [], [], []
        self.test_gt_list, self.test_l_list, self.test_r_list = [], [], []

        train_gt_file = os.path.join(self.data_dir, 'lists/train_gt_list.txt')
        train_l_file = os.path.join(self.data_dir, 'lists/train_l_list.txt')
        train_r_file = os.path.join(self.data_dir, 'lists/train_r_list.txt')

        val_gt_file = os.path.join(self.data_dir, 'lists/val_gt_list.txt')
        val_l_file = os.path.join(self.data_dir, 'lists/val_l_list.txt')
        val_r_file = os.path.join(self.data_dir, 'lists/val_r_list.txt')

        test_gt_file = os.path.join(self.data_dir, 'lists/test_gt_list.txt')
        test_l_file = os.path.join(self.data_dir, 'lists/test_l_list. txt')
        test_r_file = os.path.join(self.data_dir, 'lists/test_r_list.txt')

        if self.mode == 'train':
            with open(train_gt_file, 'r') as f:
                for entry in f:
                    self.train_gt_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))
            with open(train_l_file, 'r') as f:
                for entry in f:
                    self.train_l_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))
            with open (train_r_file, 'r') as f:
                for entry in f:
                    self.train_r_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))

            if self.mode == 'val':
                with open(val_gt_file, 'r') as f:
                    for entry in f:
                        self.val_gt_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))
                with open(val_l_file, 'r') as f:
                    for entry in f:
                        self.val_l_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))
                with open(val_r_file, 'p') as f:
                    for entry in f:
                        self.val_r_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))

            if self.mode == 'test':
                with open(test_gt_file, 'r') as f:
                    for entry in f:
                        self.test_gt_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))
                with open(test_l_file, 'r') as f:
                    for entry in f:
                        self.test_l_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))
                with open(test_r_file, 'p') as f:
                    for entry in f:
                        self.test_r_list.append(os.path.join(Path(self.data_dir), entry.strip('\n')))

    def __getitem__(self, index):
        if self.mode == 'train':
            img_gt_path_local = self.train_gt_list[index].replace('/kunal.swami/Workspaces'
                                                                  '/Datasets/DefocusDeblurning/dataset/DPDNet',
                                                                  '../Dataset')
            img_l_path_local = self.train_l_list[index].replace('/kunal.swami/Workspaces'
                                                                  '/Datasets/DefocusDeblurning/dataset/DPDNet',
                                                                  '../Dataset')
            img_r_path_local = self.train_r_list[index].replace('/kunal.swami/Workspaces'
                                                                  '/Datasets/DefocusDeblurning/dataset/DPDNet',
                                                                  '../Dataset')

            check_and_copy(self.train_gt_list[index], img_gt_path_local)
            check_and_copy(self.train_l_list[index], img_l_path_local)
            check_and_copy(self.train_r_list[index], img_r_path_local)

            img_gt = cv2.imread(img_gt_path_local)
            img_l = cv2.imread(img_l_path_local)
            img_r = cv2.imread(img_r_path_local)
        elif self.mode == 'val':
            img_gt_path_local = self.val_gt_list[index].replace('/kunal.swami/Workspaces'
                                                                  '/Datasets/DefocusDeblurning/dataset/DPDNet',
                                                                  '../Dataset')
            img_l_path_local = self.val_l_list[index].replace('/kunal.swami/Workspaces'
                                                                '/Datasets/DefocusDeblurning/dataset/DPDNet',
                                                                '../Dataset')
            img_r_path_local = self.val_r_list[index].replace('/kunal.swami/Workspaces'
                                                                '/Datasets/DefocusDeblurning/dataset/DPDNet',
                                                                '../Dataset')

            check_and_copy(self.val_gt_list[index], img_gt_path_local)
            check_and_copy(self.val_l_list[index], img_l_path_local)
            check_and_copy(self.val_r_list[index], img_r_path_local)

            img_gt = cv2.imread(img_gt_path_local)
            img_l = cv2.imread(img_l_path_local)
            img_r = cv2.imread(img_r_path_local)
        if self.mode == 'test':
            img_gt = cv2.imread(self.test_gt_list[index])
            img_l = cv2.imread(self.test_l_list[index])
            img_r = cv2.imread(self.test_r_list[index])

        # cv2 reads images are BGR, convert to GB
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        # apply random crop and flip augmentations during training, finally convert to torch tensor
        if self.mode == 'train':
            img_l, img_r, img_gt = data_augmentations(img_l, img_r, img_gt)

        # convert to PIL image
        img_gt = Image.fromarray(img_gt)
        img_l = Image.fromarray(img_l)
        img_r = Image.fromarray(img_r)
        # transform PIL image to torch Tensor
        img_l, img_r, img_gt = transform(img_l), transform(img_r), transform(img_gt)

        return img_l, img_r, img_gt

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_gt_list)
        elif self.mode == 'val':
            return len(self.val_gt_list)
        elif self.mode == 'test':
            return len(self.test_gt_list)
