

import builtins
import torch.distributed as dist
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import time
import numpy as np
import math
from torch.utils.data import DataLoader
import model.ResNet as models
from model.CaCo import CaCo, CaCo_PN
from ops.os_operation import mkdir, mkdir_rank
from training.train_utils import adjust_learning_rate2,save_checkpoint
from data_processing.loader import TwoCropsTransform, TwoCropsTransform2,GaussianBlur,Solarize
from ops.knn_monitor import knn_monitor
import torch.optim as optim
def init_log_path(args,batch_size):
    """
    :param args:
    :return:
    save model+log path
    """
    save_path = os.path.join(os.getcwd(), args.log_path)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, args.dataset)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "Type_"+str(args.type))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "lr_" + str(args.lr) + "_" + str(args.lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memlr_"+str(args.memory_lr) +"_"+ str(args.memory_lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "t_" + str(args.moco_t) + "_memt" + str(args.mem_t))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "wd_" + str(args.weight_decay) + "_memwd" + str(args.mem_wd)) 
    mkdir_rank(save_path,args.rank)
    if args.moco_m_decay:
        save_path = os.path.join(save_path, "mocomdecay_" + str(args.moco_m))
    else:
        save_path = os.path.join(save_path, "mocom_" + str(args.moco_m))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memgradm_" + str(args.mem_momentum))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "hidden" + str(args.mlp_dim)+"_out"+str(args.moco_dim))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "batch_" + str(batch_size))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "epoch_" + str(args.epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "warm_" + str(args.warmup_epochs))
    mkdir_rank(save_path,args.rank)
    return save_path

def main_worker(args):
    params = vars(args)
    print(vars(args))
    init_lr = args.lr * args.batch_size / 256
    total_batch_size = args.batch_size
    print("init lr",init_lr," init batch size",args.batch_size)
    # create model
    print("=> creating model '{}'".format(args.arch))

    Memory_Bank = CaCo_PN(args.cluster,args.moco_dim)

    model = CaCo(models.__dict__[args.arch], args,
                           args.moco_dim, args.moco_m)
    

    
    #optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                #momentum=args.momentum,
                                #weight_decay=args.weight_decay)
 
    from model.optimizer import  AdamW
    from model.optimizer import  LARS
    #optimizer = AdamW(model.parameters())
    optimizer = LARS(model.parameters(), init_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)
    #optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                #momentum=args.momentum,
                                #weight_decay=args.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    model.cuda()
    Memory_Bank.cuda()
    print("per gpu batch size: ",args.batch_size)
    print("current workers:",args.workers)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    save_path = init_log_path(args,total_batch_size)
    if not args.resume:
        args.resume = os.path.join(save_path,"checkpoint_best.pth.tar")
        print("searching resume files ",args.resume)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Memory_Bank.load_state_dict(checkpoint['Memory_Bank'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset=='ImageNet':
        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        if args.multi_crop:
            from data_processing.MultiCrop_Transform import Multi_Transform
            multi_transform = Multi_Transform([32, 24],
                                              [2, 2],
                                              [1.0, 0.5],
                                              [1.0, 1.0], normalize)
            train_dataset = datasets.ImageFolder(
                traindir, multi_transform)
        else:

            augmentation1 = [
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize
                ]

            augmentation2 = [
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize
                ]
            train_dataset = datasets.ImageFolder(
                    traindir,
                    TwoCropsTransform2(transforms.Compose(augmentation1),
                                       transforms.Compose(augmentation2)))
            
        testdir = os.path.join(args.data, 'val')
        transform_test = transforms.Compose([
            
            
            transforms.ToTensor(),
            normalize,
        ])
        from data_processing.imagenet import imagenet
        val_dataset = datasets.ImageFolder(traindir, transform_test)
        test_dataset = datasets.ImageFolder(testdir, transform_test)

    else:
        print("We only support ImageNet dataset currently")
        exit()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.knn_batch_size,pin_memory=True,num_workers=args.workers,drop_last=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.knn_batch_size,pin_memory=True,num_workers=args.workers,drop_last=False)

    #init weight for memory bank
    bank_size=args.cluster
    print("finished the data loader config!")
    model.eval()
    print("gpu consuming before running:", torch.cuda.memory_allocated()/1024/1024)
    #init memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        from training.init_memory import init_memory
        init_memory(train_loader, model, Memory_Bank, criterion,
              optimizer, 0, args)
        print("Init memory bank finished!!")
    knn_path = os.path.join(save_path,"knn.log")
    train_log_path = os.path.join(save_path,"train.log")
    best_Acc=0
    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate2(optimizer, epoch, args, init_lr)    
        #if args.type<10:
        if args.moco_m_decay:
            moco_momentum = adjust_moco_momentum(epoch, args)
        else:
            moco_momentum = args.moco_m
        print("current moco momentum %f"%moco_momentum)
        # train for one epoch
        
        from training.train_caco import train_caco
        acc1 = train_caco(train_loader, model, Memory_Bank, criterion,
                                optimizer, epoch, args, train_log_path,moco_momentum)

        if epoch%args.knn_freq==0 or epoch<=20:
            print("gpu consuming before cleaning:", torch.cuda.memory_allocated()/1024/1024)
            torch.cuda.empty_cache()
            print("gpu consuming after cleaning:", torch.cuda.memory_allocated()/1024/1024)

            try:
                knn_test_acc=knn_monitor(model.encoder_q, val_loader, test_loader,
                        global_k=min(args.knn_neighbor,len(val_loader.dataset)))
                print({'*KNN monitor Accuracy': knn_test_acc})
                if args.rank ==0:
                    with open(knn_path,'a+') as file:
                        file.write('%d epoch KNN monitor Accuracy %f\n'%(epoch,knn_test_acc))
            except:
                print("small error raised in knn calcu")
                knn_test_acc=0

            torch.cuda.empty_cache()
            epoch_limit=20
            if knn_test_acc<=1.0 and epoch>=epoch_limit:
                exit()
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)

        save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            'knn_acc': knn_test_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }


        if epoch%10==9:
            tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
        tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')

        save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
