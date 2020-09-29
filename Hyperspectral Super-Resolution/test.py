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

parser = argparse.ArgumentParser(description='HyRISR Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--network', default='Hyperspectral_RCAN', type=str,
                    help='network')
parser.add_argument('--lr-image-size', default=16, type=int,
                    help='low resolution image size (default: 16)')
parser.add_argument('--hr-image-size', default=64, type=int,
                    help='high resolution image size (default: 64)')
parser.add_argument('--spectrum-len', default=500, type=int,
                    help='spectrum length (default: 500)')
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
    scale = args.hr_image_size // args.lr_image_size
    net = model.Hyperspectral_RCAN(args.spectrum_len, scale).float()

    if scale == 2:
        net.load_state_dict(torch.load('RCAN_2x.pt'))
    elif scale == 3:
        net.load_state_dict(torch.load('RCAN_3x.pt'))
    else: #scale == 4
        net.load_state_dict(torch.load('RCAN_4x.pt'))


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
        net = nn.DataParallel(net).cuda()
       
    # ----------------------------------------------------------------------------------------
    # Define dataset path and data splits
    # ----------------------------------------------------------------------------------------    
    dataset_path = "\Path\To\Dataset\"
    image_ids_csv = pd.read_csv(dataset_path + "Image_IDs.csv")

    image_ids = image_ids_csv["id"].values

    # ----------------------------------------------------------------------------------------
    # Create datasets and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Test = dataset.RamanImageDataset(image_ids, dataset_path, batch_size = args.batch_size, 
                                                    hr_image_size = args.hr_image_size, lr_image_size = args.lr_image_size,
                                                    spectrum_len = args.spectrum_len)

    test_loader = DataLoader(Raman_Dataset_Test, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)

    # ----------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------
    RCAN_PSNR, Bicubic_PSNR, Nearest_PSNR, RCAN_SSIM, Bicubic_SSIM, Nearest_SSIM, RCAN_MSE, Bicubic_MSE, Nearest_MSE = evaluate(test_loader, net, scale, args)

def evaluate(dataloader, net, scale, args):
    
    psnr = utilities.AverageMeter('PSNR', ':.4f')
    ssim = utilities.AverageMeter('SSIM', ':.4f')
    mse_NN = utilities.AverageMeter('MSE', ':.4f')
    psnr_bicubic = utilities.AverageMeter('PSNR_Bicubic', ':.4f')
    ssim_bicubic = utilities.AverageMeter('SSIM_Bicubic', ':.4f')
    mse_bicubic = utilities.AverageMeter('MSE_Bicubic', ':.4f')
    psnr_nearest_neighbours = utilities.AverageMeter('PSNR_Nearest_Neighbours', ':.4f')
    ssim_nearest_neighbours = utilities.AverageMeter('SSIM_Nearest_Neighbours', ':.4f')
    mse_nearest_neighbours = utilities.AverageMeter('MSE_Nearest_Neighbours', ':.4f')
    
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # measure data loading time
            x = data['input_image']
            inputs = x.float()
            inputs = inputs.cuda(args.gpu)
            y = data['output_image']
            target = y.float()
            target = target.cuda(args.gpu)

            # compute output
            output = net(inputs)

            x2 = np.squeeze(x.numpy())
            y2 = np.squeeze(y.numpy())

            nearest_neighbours = scipy.ndimage.zoom(x2,(1,scale,scale), order=0)
            bicubic = scipy.ndimage.zoom(x2,(1,scale,scale), order=3)
                            
            bicubic = torch.from_numpy(bicubic)
            bicubic = bicubic.cuda(args.gpu)
            
            nearest_neighbours = torch.from_numpy(nearest_neighbours)
            nearest_neighbours = nearest_neighbours.cuda(args.gpu)

            # Nearest neighbours
            psnr_batch_nearest_neighbours = utilities.calc_psnr(nearest_neighbours, target)
            psnr_nearest_neighbours.update(psnr_batch_nearest_neighbours, inputs.size(0))

            ssim_batch_nearest_neighbours = utilities.calc_ssim(nearest_neighbours, target)
            ssim_nearest_neighbours.update(ssim_batch_nearest_neighbours, inputs.size(0))

            mse_batch_nearest_neighbours = nn.MSELoss()(nearest_neighbours, target)
            mse_nearest_neighbours.update(mse_batch_nearest_neighbours, inputs.size(0))
            
            # Bicubic
            psnr_batch_bicubic = utilities.calc_psnr(bicubic, target)
            psnr_bicubic.update(psnr_batch_bicubic, inputs.size(0))

            ssim_batch_bicubic = utilities.calc_ssim(bicubic, target)
            ssim_bicubic.update(ssim_batch_bicubic, inputs.size(0))

            mse_batch_bicubic = nn.MSELoss()(bicubic, target)
            mse_bicubic.update(mse_batch_bicubic, inputs.size(0))
            
            # Neural network
            psnr_batch = utilities.calc_psnr(output, target)
            psnr.update(psnr_batch, inputs.size(0))

            ssim_batch = utilities.calc_ssim(output, target)
            ssim.update(ssim_batch, inputs.size(0))
            
            mse_batch = nn.MSELoss()(output, target)
            mse_NN.update(mse_batch, inputs.size(0))
            
    print("RCAN PSNR: {}    Bicubic PSNR: {}    Nearest Neighbours PSNR: {}".format(psnr.avg, psnr_bicubic.avg, psnr_nearest_neighbours.avg))
    print("RCAN SSIM: {}    Bicubic SSIM: {}    Nearest Neighbours SSIM: {}".format(ssim.avg, ssim_bicubic.avg, ssim_nearest_neighbours.avg))
    print("RCAN MSE:  {}    Bicubic MSE:  {}    Nearest Neighbours MSE:  {}".format(mse_NN.avg, mse_bicubic.avg, mse_nearest_neighbours.avg))
    return psnr.avg, psnr_bicubic.avg, psnr_nearest_neighbours.avg, ssim.avg, ssim_bicubic.avg, ssim_nearest_neighbours.avg, mse_NN.avg, mse_bicubic.avg, mse_nearest_neighbours.avg

if __name__ == '__main__':
    main()