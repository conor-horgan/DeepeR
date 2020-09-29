import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class BasicConv(nn.Module):
    def __init__(self, channels_in, channels_out, batch_norm):
        super(BasicConv, self).__init__()
        basic_conv = [nn.Conv1d(channels_in, channels_out, kernel_size = 3, stride = 1, padding = 1, bias = True)]
        basic_conv.append(nn.PReLU())
        if batch_norm:
            basic_conv.append(nn.BatchNorm1d(channels_out))
        
        self.body = nn.Sequential(*basic_conv)

    def forward(self, x):
        return self.body(x)

class ResUNetConv(nn.Module):
    def __init__(self, num_convs, channels, batch_norm):
        super(ResUNetConv, self).__init__()
        unet_conv = []
        for _ in range(num_convs):
            unet_conv.append(nn.Conv1d(channels, channels, kernel_size = 3, stride = 1, padding = 1, bias = True))
            unet_conv.append(nn.PReLU())
            if batch_norm:
                unet_conv.append(nn.BatchNorm1d(channels))
        
        self.body = nn.Sequential(*unet_conv)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class UNetLinear(nn.Module):
    def __init__(self, repeats, channels_in, channels_out):
        super().__init__()
        modules = []
        for i in range(repeats):
            modules.append(nn.Linear(channels_in, channels_out))
            modules.append(nn.PReLU())
        
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, num_convs, batch_norm):
        super(ResUNet, self).__init__()
        res_conv1 = [BasicConv(1, 64, batch_norm)]
        res_conv1.append(ResUNetConv(num_convs, 64, batch_norm))
        self.conv1 = nn.Sequential(*res_conv1)
        self.pool1 = nn.MaxPool1d(2)

        res_conv2 = [BasicConv(64, 128, batch_norm)]
        res_conv2.append(ResUNetConv(num_convs, 128, batch_norm))
        self.conv2 = nn.Sequential(*res_conv2)
        self.pool2 = nn.MaxPool1d(2)

        res_conv3 = [BasicConv(128, 256, batch_norm)]
        res_conv3.append(ResUNetConv(num_convs, 256, batch_norm))
        res_conv3.append(BasicConv(256, 128, batch_norm))
        self.conv3 = nn.Sequential(*res_conv3)       
        self.up3 = nn.Upsample(scale_factor = 2)

        res_conv4 = [BasicConv(256, 128, batch_norm)]
        res_conv4.append(ResUNetConv(num_convs, 128, batch_norm))
        res_conv4.append(BasicConv(128, 64, batch_norm))
        self.conv4 = nn.Sequential(*res_conv4)     
        self.up4 = nn.Upsample(scale_factor = 2)

        res_conv5 = [BasicConv(128, 64, batch_norm)]
        res_conv5.append(ResUNetConv(num_convs,64, batch_norm))
        self.conv5 = nn.Sequential(*res_conv5)
        res_conv6 = [BasicConv(64, 1, batch_norm)]
        self.conv6 = nn.Sequential(*res_conv6)    
        
        self.linear7 = UNetLinear(3, 500, 500)
               
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x2 = self.conv2(x1)
        x3 = self.pool1(x2)
        
        x3 = self.conv3(x3)
        x3 = self.up3(x3)
        
        x4 = torch.cat((x2, x3), dim = 1)
        x4 = self.conv4(x4)
        x5 = self.up4(x4)

        x6 = torch.cat((x, x5), dim = 1)
        x6 = self.conv5(x6)
        x7 = self.conv6(x6)
        
        out = self.linear7(x7)
        
        return out