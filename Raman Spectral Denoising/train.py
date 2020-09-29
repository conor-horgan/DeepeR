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
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 5e-4)', dest='lr')
parser.add_argument('--base-lr', '--base-learning-rate', default=5e-6, type=float,
                    help='base learning rate (default: 5e-6)')
parser.add_argument('--scheduler', default='constant-lr', type=str,
                    help='scheduler')
parser.add_argument('--batch-norm', action='store_true',
                    help='apply batch norm')
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
    net = model.ResUNet(3, args.batch_norm).float()

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

    Input = Input_Data['Inputs']
    Output = Output_Data['Outputs']

    spectra_num = len(Input)

    train_split = round(0.9 * spectra_num)
    val_split = round(0.1 * spectra_num)

    input_train = Input[:train_split]
    input_val = Input[train_split:train_split+val_split]

    output_train = Output[:train_split]
    output_val = Output[train_split:train_split+val_split]

    # ----------------------------------------------------------------------------------------
    # Create datasets (with augmentation) and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Train = dataset.RamanDataset(input_train, output_train, batch_size = args.batch_size, spectrum_len = args.spectrum_len,
                                   spectrum_shift=0.1, spectrum_window = False, horizontal_flip = False, mixup = True)

    Raman_Dataset_Val = dataset.RamanDataset(input_val, output_val, batch_size = args.batch_size, spectrum_len = args.spectrum_len)

    train_loader = DataLoader(Raman_Dataset_Train, batch_size = args.batch_size, shuffle = False, num_workers = 0, pin_memory = True)
    val_loader = DataLoader(Raman_Dataset_Val, batch_size = args.batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    # ----------------------------------------------------------------------------------------
    # Define criterion(s), optimizer(s), and scheduler(s)
    # ----------------------------------------------------------------------------------------
    criterion = nn.L1Loss().cuda(args.gpu)
    criterion_MSE = nn.MSELoss().cuda(args.gpu)
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr = args.lr)
    elif args.optimizer == "adamW":
        optimizer = optim.AdamW(net.parameters(), lr = args.lr)
    else: # Adam
        optimizer = optim.Adam(net.parameters(), lr = args.lr)

    if args.scheduler == "decay-lr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    elif args.scheduler == "multiplicative-lr":
        lmbda = lambda epoch: 0.985
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif args.scheduler == "cyclic-lr":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = args.base_lr, max_lr = args.lr, mode = 'triangular2', cycle_momentum = False)
    elif args.scheduler == "one-cycle-lr":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, cycle_momentum = False)
    else: # constant-lr
        scheduler = None

    print('Started Training')
    print('Training Details:')
    print('Network:         {}'.format(args.network))
    print('Epochs:          {}'.format(args.epochs))
    print('Batch Size:      {}'.format(args.batch_size))
    print('Optimizer:       {}'.format(args.optimizer))
    print('Scheduler:       {}'.format(args.scheduler))
    print('Learning Rate:   {}'.format(args.lr))
    print('Spectrum Length: {}'.format(args.spectrum_len))

    DATE = datetime.datetime.now().strftime("%Y_%m_%d")

    log_dir = "runs/{}_{}_{}_{}".format(DATE, args.optimizer, args.scheduler, args.network)
    models_dir = "{}_{}_{}_{}.pt".format(DATE, args.optimizer, args.scheduler, args.network)

    writer = SummaryWriter(log_dir = log_dir)

    for epoch in range(args.epochs):
        train_loss = train(train_loader, net, optimizer, scheduler, criterion, criterion_MSE, epoch, args)
        val_loss = validate(val_loader, net, criterion_MSE, args)
        if args.scheduler == "decay-lr" or args.scheduler == "multiplicative-lr":
            scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
               
    torch.save(net.state_dict(), models_dir)
    print('Finished Training')

def train(dataloader, net, optimizer, scheduler, criterion, criterion_MSE, epoch, args):
    
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    progress = utilities.ProgressMeter(len(dataloader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, data in enumerate(dataloader):
        inputs = data['input_spectrum']
        inputs = inputs.float()
        inputs = inputs.cuda(args.gpu)
        target = data['output_spectrum']
        target = target.float()
        target = target.cuda(args.gpu)

        output = net(inputs)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if args.scheduler == "cyclic-lr" or args.scheduler == "one-cycle-lr":
            scheduler.step()   

        loss_MSE = criterion_MSE(output, target)
        losses.update(loss_MSE.item(), inputs.size(0)) 

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 400 == 0:
            progress.display(i)
    return losses.avg


def validate(dataloader, net, criterion_MSE, args):
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    progress = utilities.ProgressMeter(len(dataloader), [batch_time, losses], prefix='Validation: ')

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(dataloader):
            inputs = data['input_spectrum']
            inputs = inputs.float()
            inputs = inputs.cuda(args.gpu)
            target = data['output_spectrum']
            target = target.float()
            target = target.cuda(args.gpu)

            output = net(inputs)

            loss_MSE = criterion_MSE(output, target)
            losses.update(loss_MSE.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 400 == 0:
                progress.display(i)

    return losses.avg


if __name__ == '__main__':
    main()