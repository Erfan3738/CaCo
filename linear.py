
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import model.ResNet as models
import torchvision.models as models
from ops.os_operation import mkdir
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))+['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default= 1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument("--dataset", type=str, default="ImageNet", help="which dataset is used to finetune")

best_acc1 = 0


def main():
    args = parser.parse_args()


    main_worker(args)


def main_worker(args):
    global best_acc1
    

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 10
    model = models.__dict__[args.arch](num_classes=num_classes)
    #model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    #model.maxpool = nn.Identity()
    
    # freeze all layers but the last fc
    #for name, param in model.named_parameters():
        #if name not in ['fc.weight', 'fc.bias']:
            #param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    linear_keyword="fc"
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename caco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    del state_dict[k]
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))  

    print(model)
    


    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256


    model.cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.pretrained = os.path.abspath(args.pretrained)
    save_dir = os.path.split(args.pretrained)[0]
    if args.rank==0:
        mkdir(save_dir)
    save_dir = os.path.join(save_dir, "linear_lars")
    if args.rank==0:
        mkdir(save_dir)
    save_dir = os.path.join(save_dir, "bs_%d" % args.batch_size)
    if args.rank==0:
        mkdir(save_dir)
    save_dir = os.path.join(save_dir, "lr_%.3f" % args.lr)
    if args.rank==0:
        mkdir(save_dir)
    save_dir = os.path.join(save_dir, "wd_" + str(args.weight_decay))
    if args.rank==0:
        mkdir(save_dir)
    # optimize only the linear classifier
    #parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    #assert len(parameters) == 2  # fc.weight, fc.bias
    init_lr = args.lr
    #optimizer = torch.optim.Adam(model.parameters(), init_lr, betas=(0.9, 0.999),
                 #eps=1e-08, weight_decay=1e-4, amsgrad=True)
    from model.optimizer import  LARS
    
    optimizer = LARS(model.parameters(), init_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)


    #optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                #momentum=args.momentum,
                                #weight_decay=args.weight_decay)
    print("=> use LARS optimizer.")
    from ops.LARS import SGD_LARC
    #optimizer = SGD_LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)
    # optionally resume from a checkpoint

    if args.resume is None:
        args.resume = os.path.join(save_dir,"checkpoint.pth.tar")
    if args.resume is not None and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        best_acc1 =0

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    augmentation1 = transforms.Compose([
                
                transforms.Resize(224, antialias=True),
                #transforms.RandomCrop(224, padding=4),
                #transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                    
                #transforms.RandomApply([
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)  # not strengthened
                    #], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                    #transforms.GaussianBlur(kernel_size=5),
                transforms.ToTensor(),
                normalize
                ])
    train_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=augmentation1)

    transform_test = transforms.Compose([
        transforms.Resize(224, antialias=True),              # Resize to 256x256
        #transforms.CenterCrop(224),
        
        
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset =CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,drop_last=False)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    log_path = save_dir
    acc_path = os.path.join(log_path,"acc.log")
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        with open(acc_path,'a+') as file:
            file.write('%d epoch  Accuracy %f\n'%(epoch,acc1))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            tmp_save_path = os.path.join(log_path, 'checkpoint.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=tmp_save_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        
        #print("Images shape:", images.shape)  # Should be (batch_size, 3, 32, 32)
        #print("Target shape:", target.shape)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        
        #print("Model output shape:", output.shape)  # Expected: (1024, 10)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # print(images.size())
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        root_path = os.path.split(filename)[0]
        best_path = os.path.join(root_path, "model_best.pth.tar")
        shutil.copyfile(filename, best_path)



def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k[len(''):] \
            if k.startswith('') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
