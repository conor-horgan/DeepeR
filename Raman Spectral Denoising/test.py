import os
import sys
import random
import datetime
import time
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import math
from skimage.measure import compare_ssim as sk_ssim

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

import model, dataset, utilities


parser = argparse.ArgumentParser(description='DeNoiser Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--spectrum-len', default=500, type=int,
                    help='spectrum length (default: 350)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # ----------------------------------------------------------------------------------------
    # Create model(s) and send to device(s)
    # ----------------------------------------------------------------------------------------
    net = model.ResUNet(3, False).float()
    net.load_state_dict(torch.load('ResUNet_Fold_7.pt'))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            net.cuda(args.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        else:
            net.cuda(args.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)
    else:
        net.cuda(args.gpu)
        net = torch.nn.parallel.DistributedDataParallel(net)

    # ----------------------------------------------------------------------------------------
    # Define dataset path and data splits
    # ----------------------------------------------------------------------------------------    
    Input_Data = scipy.io.loadmat("\Path\To\Inputs.mat")
    Output_Data = scipy.io.loadmat("\Path\To\Outputs.mat")

    Input = Input_Data['data']
    Output = Output_Data['data']

    # ----------------------------------------------------------------------------------------
    # Create datasets (with augmentation) and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Test = dataset.RamanDataset(Input, Output, batch_size = args.batch_size, spectrum_len = args.spectrum_len)

    test_loader = DataLoader(Raman_Dataset_Test, batch_size = args.batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    # ----------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------
    MSE_NN, MSE_SG = evaluate(test_loader, net, args)

def evaluate(dataloader, net, args):
    losses = utilities.AverageMeter('Loss', ':.4e')
    SG_loss = utilities.AverageMeter('Savitzky-Golay Loss', ':.4e')

    net.eval()
    
    MSE_SG = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x = data['input_spectrum']
            inputs = x.float()
            inputs = inputs.cuda(args.gpu)
            y = data['output_spectrum']
            target = y.float()
            target = target.cuda(args.gpu)
            
            x = np.squeeze(x.numpy())
            y = np.squeeze(y.numpy())

            output = net(inputs)
            loss = nn.MSELoss()(output, target)
            
            x_out = output.cpu().detach().numpy()
            x_out = np.squeeze(x_out)

            SGF_1_9 = scipy.signal.savgol_filter(x,9,1)
            MSE_SGF_1_9 = np.mean(np.mean(np.square(np.absolute(y - (SGF_1_9 - np.reshape(np.amin(SGF_1_9, axis = 1), (len(SGF_1_9),1)))))))
            MSE_SG.append(MSE_SGF_1_9)
            
            losses.update(loss.item(), inputs.size(0))

        print("Neural Network MSE: {}".format(losses.avg))
        print("Savitzky-Golay MSE: {}".format(np.mean(np.asarray(MSE_SG))))
        print("Neural Network performed {0:.2f}x better than Savitzky-Golay".format(np.mean(np.asarray(MSE_SG))/losses.avg))

    return losses.avg, MSE_SG

if __name__ == '__main__':
    main()