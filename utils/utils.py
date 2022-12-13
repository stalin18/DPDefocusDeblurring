import os
import collections
import shutil

import numpy as np

import torch
from torch import nn


class AverageMeter(object):
    """ Computes and stores the average and count """

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, weight=1):
        if isinstance(val, collections.Iterable):
            if not isinstance(weight, collections.Iterable):
                weight = np.ones(len(val))
            for i in range(len(val)):
                self.sum += val[i] * weight[i]
                self.count += weight[i]
        else:
            self.sum += val
            self.count += weight
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

    def count(self):
        return self.count


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))

        return loss


def check_and_copy(src_pth, dst_pth):
    if not os.path.exists(dst_pth):
        os.makedirs(os.path.dirname(dst_pth), exist_ok=True)
        shutil.copy(src_pth, dst_pth)
