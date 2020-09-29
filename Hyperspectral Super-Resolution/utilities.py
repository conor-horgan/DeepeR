import os
import sys
import random
import datetime
import time
import shutil

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import math
from skimage.measure import compare_ssim as sk_ssim

import torch
from torch import nn

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def calc_psnr(output, target):
    psnr = 0.
    mse = nn.MSELoss()(output, target)
    psnr = 10 * math.log10(torch.max(output)/mse)
    return psnr

def calc_ssim(output, target):
    ssim = 0.
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    if output.ndim == 4:
        for i in range(output.shape[0]):
            output_i = np.squeeze(output[i,:,:,:])
            output_i = np.moveaxis(output_i, 0, -1)
            target_i = np.squeeze(target[i,:,:,:])
            target_i = np.moveaxis(target_i, 0, -1)
            batch_size = output.shape[0]
            ssim += sk_ssim(output_i, target_i, data_range = output_i.max() - target_i.max(), multichannel=True)
    else:
        output_i = np.squeeze(output)
        output_i = np.moveaxis(output_i, 0, -1)
        target_i = np.squeeze(target)
        target_i = np.moveaxis(target_i, 0, -1)
        batch_size = 1
        ssim += sk_ssim(output_i, target_i, data_range = output_i.max() - target_i.max(), multichannel=True)
        
    ssim = ssim / batch_size
    return ssim