

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
from training.train_utils import adjust_learning_rate2,save_checkpoint,adjust_learning_rate
from data_processing.loader import TwoCropsTransform, TwoCropsTransform2,GaussianBlur,Solarize
from ops.knn_monitor import knn_monitor
import torch.optim as optim
from torchvision.datasets import CIFAR10, STL10,Imagenette 
from torch.optim.optimizer import Optimizer

import torch
from torch.optim.optimizer import Optimizer


import torch
from torch.optim.optimizer import Optimizer


class EnhancedSGD(Optimizer):
    """Enhanced SGD implementation with:
    1. Stochastic Weight Averaging (SWA)
    2. Gradient clipping/normalization
    3. Selective weight decay (excluding biases and BatchNorm)
    4. Momentum with optional Nesterov acceleration
    
    Learning rate scheduling should be handled externally.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): base learning rate
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 1e-4)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: True)
        exclude_bn_bias (bool, optional): exclude BN and bias from weight decay (default: True)
        clip_grad_norm (float, optional): max norm of gradients (default: None)
        swa_start (int): epoch to start SWA averaging
        swa_freq (int): frequency of SWA model updates
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 dampening=0, nesterov=True, exclude_bn_bias=True,
                 clip_grad_norm=None, swa_start=None, swa_freq=5):
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        clip_grad_norm=clip_grad_norm)
        
        super(EnhancedSGD, self).__init__(params, defaults)
        
        self.exclude_bn_bias = exclude_bn_bias
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        
        # Initialize SWA state
        if swa_start is not None:
            self.swa_state = {
                'models_count': 0,
                'swa_model': None
            }
        
        # Identify parameters to exclude from weight decay
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                
                # Check if parameter is a bias or from BatchNorm
                is_batch_norm = False
                is_bias = False
                
                if hasattr(p, 'name') and p.name:
                    is_batch_norm = 'bn' in p.name.lower() or 'batch_norm' in p.name.lower()
                    is_bias = 'bias' in p.name.lower()
                elif p.dim() == 1:
                    is_bias = True
                
                param_state['apply_weight_decay'] = not (is_bias or is_batch_norm)

    def step(self, closure=None, epoch=None, model=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
            epoch (int, optional): Current epoch number for SWA updates
            model (nn.Module, optional): Model for SWA
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Update SWA model if in SWA phase
        if self.swa_start is not None and epoch is not None and model is not None:
            if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
                self._update_swa_model(model)
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            clip_norm = group['clip_grad_norm']
            
            # Apply gradient clipping if specified
            if clip_norm is not None:
                parameters = [p for p in group['params'] if p.grad is not None]
                if parameters:
                    torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                param_state = self.state[p]
                
                # Apply weight decay selectively
                if weight_decay != 0 and param_state.get('apply_weight_decay', True):
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                p.data.add_(d_p, alpha=-group['lr'])
        
        
    
    def _update_swa_model(self, model):
        """Updates the SWA model using the current model weights"""
        if 'swa_model' not in self.swa_state or self.swa_state['swa_model'] is None:
            # Initialize SWA model by copying the current model
            self.swa_state['swa_model'] = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }
            self.swa_state['models_count'] = 1
        else:
            # Update SWA model with moving average
            model_count = self.swa_state['models_count']
            for name, param in model.named_parameters():
                if name in self.swa_state['swa_model']:
                    self.swa_state['swa_model'][name].mul_(model_count / (model_count + 1.0))
                    self.swa_state['swa_model'][name].add_(param.data, alpha=1.0 / (model_count + 1.0))
            self.swa_state['models_count'] += 1
    
    def swap_swa_sgd(self, model):
        """Swaps the model parameters with the averaged SWA parameters"""
        if not hasattr(self, 'swa_state') or self.swa_state['swa_model'] is None:
            print("SWA model has not been initialized yet.")
            return
        
        # Transfer SWA weights to model
        for name, param in model.named_parameters():
            if name in self.swa_state['swa_model']:
                param.data.copy_(self.swa_state['swa_model'][name])
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
    print(model.encoder_q)
    #optimizer, scheduler = setup_optimizer_with_no_lr_scheduler_for_projection_head(model)

    
    #optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                #momentum=args.momentum,
                                #weight_decay=args.weight_decay)
 
    from model.optimizer import  AdamW
    #from model.optimizer import  LARS
    #optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    #optimizer = LARS(model.parameters(), args.lr ,weight_decay=args.weight_decay,momentum=args.momentum)
    
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, args.weight_decay, args.momentum)

optimizer = EnhancedSGD(
    model.parameters(),
    args.lr,                # Initial learning rate
    args.momentum,          # Momentum coefficient
    args.weight_decay,     # Weight decay
    clip_grad_norm=1.0,    # Max gradient norm
    swa_start=100,          # Start SWA from epoch 75
    swa_freq=5             # Update SWA model every 5 epochs
)


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
                checkpoint = torch.load(args.resume, map_location=loc,weights_only=False)
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
    if args.dataset=='stl10':
        #traindir = os.path.join(args.data, 'train')
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

            augmentation1 = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    #transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                    
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize
                ])

            augmentation2 = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                    #transforms.RandomApply([Solarize()], p=0.1),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize
                ])
            
    
                                    
            train_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=TwoCropsTransform(augmentation1))
            #train_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
            #train_dataset = STL10(root='./data', split='unlabeled', download=True, transform=TwoCropsTransform2(augmentation1, augmentation2))
            #train_dataset = Imagenette(root =  './data', split= 'train', size= 'full', download=True, transform =TwoCropsTransform2(augmentation1, augmentation2))
            
        testdir = os.path.join(args.data, 'val')
        transform_test = transforms.Compose([
            
            #transforms.Resize(32),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        from data_processing.imagenet import imagenet
        val_dataset =CIFAR10(root='./datasets', train=True, download=True, transform=transform_test)
        #val_dataset = STL10(root='./data', split='train', download=True, transform=transform_test)
        #val_dataset= Imagenette(root =  './data/val', split= 'train', size= 'full', download=True, transform =transform_test)
        
        test_dataset =CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
        #test_dataset = STL10(root='./data', split='test', download=True, transform=transform_test
        #test_dataset = Imagenette(root =  './data/test', split= 'val', size= 'full', download=True, transform =transform_test)

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

        adjust_learning_rate(optimizer, epoch, args)
        #adjust_learning_rate2(optimizer, epoch, args, args.lr)
        #scheduler.step()
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

        if epoch%args.knn_freq==0 or epoch<=20 or epoch==621:
            print("gpu consuming before cleaning:", torch.cuda.memory_allocated()/1024/1024)
            torch.cuda.empty_cache()
            print("gpu consuming after cleaning:", torch.cuda.memory_allocated()/1024/1024)
            acc=knn_monitor(model.encoder_q, val_loader, test_loader,epoch, args,global_k = args.knn_neighbor) 
            print({'*KNN monitor Accuracy': acc})
            if args.rank ==0:
                    with open(knn_path,'a+') as file:
                        file.write('%d epoch KNN monitor Accuracy %f\n'%(epoch,acc))
            
                                         
                        #global_k=min(args.knn_neighbor,len(val_loader.dataset))
            
            #except:
                #print("small error raised in knn calcu")
                #knn_test_acc=0

            torch.cuda.empty_cache()
            epoch_limit=20
            if acc<=1.0 and epoch>=epoch_limit:
                exit()
        is_best=best_Acc>acc
        best_Acc=max(best_Acc,acc)

        save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            'knn_acc': acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }


        if epoch%10==9:
            tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            torch.save(model, os.path.join(save_path, 'my_model.pth'))
            save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
        tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')

        save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
