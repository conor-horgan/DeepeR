import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# This code is adapted from: https://github.com/yulunzhang/RCAN
# The corresponding paper is:
# Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu, 
# Image Super-Resolution Using Very Deep Residual Channel Attention Networks
# ECCV, 2018
# Available from: https://arxiv.org/abs/1807.02758

class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.chan_attn = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.chan_attn(y)
        return x * y

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels=500, kernel_size=3, reduction=16, bias=True, act=nn.ReLU(True)):
        super(ResidualChannelAttentionBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size//2), bias=bias))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttentionBlock(channels, reduction))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, channels=500, kernel_size=3, reduction=16, bias=True, act=nn.ReLU(True), n_resblocks=6):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [ResidualChannelAttentionBlock(channels, kernel_size, reduction, bias=bias, act=nn.ReLU(True)) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size//2), bias=bias))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, channels, kernel_size, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                conv = nn.Conv2d(channels, 4*channels, kernel_size, padding=(kernel_size//2), bias=bias)
                m.append(conv)
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(channels))
                if act: m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(nn.Conv2d(channels, 9*channels, kernel_size, padding=(kernel_size//2), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(channels))
            if act: m.append(nn.ReLU(True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Hyperspectral_RCAN(nn.Module):
    def __init__(self, spectrum_len, scale=4, kernel_size=3, reduction=16, bias=True, act=nn.ReLU(True), n_resblocks=16, n_resgroups=18):
        super(Hyperspectral_RCAN, self).__init__()       
        modules_head1 = [Upsampler(scale, spectrum_len, kernel_size, act=False), nn.Conv2d(spectrum_len, spectrum_len, kernel_size, padding=(kernel_size//2), bias=bias)]
        modules_head2 = [nn.Conv2d(spectrum_len, int(spectrum_len/2), kernel_size, padding=(kernel_size//2), bias=bias)]

        modules_body = [ResidualGroup(int(spectrum_len/2), kernel_size, reduction, act, n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(int(spectrum_len/2), int(spectrum_len/2), kernel_size, padding=(kernel_size//2), bias=bias))

        modules_tail = [nn.Conv2d(int(spectrum_len/2), int(spectrum_len/2), kernel_size, padding=(kernel_size//2), bias=bias)]
        modules_tail.append(nn.Conv2d(int(spectrum_len/2), spectrum_len, kernel_size, padding=(kernel_size//2), bias=bias))

        self.head1 = nn.Sequential(*modules_head1)
        self.head2 = nn.Sequential(*modules_head2)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head1(x)
        x1 = self.head2(x)

        res1 = self.body(x1)
        res1 += x1

        res2 = self.tail(res1)
        res2 += x

        return res2