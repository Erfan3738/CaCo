
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitBatchNorm1d(nn.BatchNorm1d):
    """
    A custom BatchNorm1d implementation that splits each batch into multiple
    parts, computes statistics separately on each split, and averages them.
    
    Args:
        num_features (int): Number of features (C) in the input
        num_splits (int): Number of splits to divide the batch into
        **kwargs: Additional arguments for the parent BatchNorm1d class
    """
    def __init__(self, num_features, num_splits=2, **kwargs):
        super(SplitBatchNorm1d, self).__init__(num_features, **kwargs)
        self.num_splits = num_splits
        
    def forward(self, input):
        # Get the batch size
        N, C = input.shape[0], input.shape[1]
        
        # Check if batch size is divisible by num_splits
        if N % self.num_splits != 0:
            # If not divisible, adjust the split size to make it work
            split_size = N // self.num_splits
            remainder = N % self.num_splits
            if split_size == 0:
                # If batch size is smaller than num_splits, just use the original BN
                return super(SplitBatchNorm1d, self).forward(input)
        else:
            split_size = N // self.num_splits
            remainder = 0
        
        # If in training mode, compute custom statistics
        if self.training or not self.track_running_stats:
            # Initialize accumulators for mean and variance
            running_mean = torch.zeros(C, dtype=input.dtype, device=input.device)
            running_var = torch.zeros(C, dtype=input.dtype, device=input.device)
            
            # Process each split
            processed = 0
            for i in range(self.num_splits):
                # Handle the last split which might have a different size
                if i == self.num_splits - 1 and remainder > 0:
                    current_split_size = split_size + remainder
                else:
                    current_split_size = split_size
                
                # Extract the current split
                split_input = input[processed:processed+current_split_size]
                processed += current_split_size
                
                # Skip empty splits
                if current_split_size == 0:
                    continue
                
                # Calculate mean and variance for this split
                split_mean = split_input.mean([0])
                split_var = split_input.var([0], unbiased=False)
                
                # Accumulate statistics
                running_mean += split_mean
                running_var += split_var
            
            # Average the statistics
            mean = running_mean / self.num_splits
            var = running_var / self.num_splits
            
            # Update running statistics if tracking
            if self.training and self.track_running_stats:
                with torch.no_grad():  # Prevent gradient computation for running stats
                    momentum = self.momentum if self.momentum is not None else 0.1
                    self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
                    self.running_var = (1 - momentum) * self.running_var + momentum * var
            
            # Normalize the input
            input_normalized = (input - mean[None, :]) / torch.sqrt(var[None, :] + self.eps)
            
            # Apply scaling and shifting
            if self.affine:
                input_normalized = input_normalized * self.weight[None, :] + self.bias[None, :]
                
            return input_normalized
        else:
            # In evaluation mode, use running statistics
            input_normalized = (input - self.running_mean[None, :]) / torch.sqrt(self.running_var[None, :] + self.eps)
            
            if self.affine:
                input_normalized = input_normalized * self.weight[None, :] + self.bias[None, :]
                
            return input_normalized

class CaCo(nn.Module):
   
    def __init__(self, base_encoder,args, dim=128, m=0.99):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CaCo, self).__init__()
        self.args=args
        self.m = m
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_q.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder_q.maxpool = nn.Identity()
        #self.encoder_q.layer2[0].conv1.stride = (1, 1)
        #self.encoder_q.layer2[0].downsample[0].stride = (1, 1) if hasattr(self.encoder_q.layer2[0], 'downsample') else None

        
        self.encoder_k = base_encoder(num_classes=dim)
        self.encoder_k.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder_k.maxpool = nn.Identity()
        #self.encoder_k.layer2[0].conv1.stride = (1, 1)
        #self.encoder_k.layer2[0].downsample[0].stride = (1, 1) if hasattr(self.encoder_k.layer2[0], 'downsample') else None
        
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        print("dim_mlp:",dim_mlp)
        # we do not keep 
        #self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        self.encoder_q.fc = self._build_mlp(2,dim_mlp,args.mlp_dim,dim,last_bn=False,num_splits=8)
        self.encoder_k.fc = self._build_mlp(2,dim_mlp,args.mlp_dim,dim,last_bn=False,num_splits=8)
        
        #self.encoder_q.fc = self._build_mlp(2,dim_mlp,args.mlp_dim,dim,last_bn=True)
        #self.encoder_k.fc = self._build_mlp(2, dim_mlp, args.mlp_dim, dim, last_bn=True)
        #self.predictor = self._build_mlp(2,dim,args.mlp_dim,dim,last_bn=False)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K=args.cluster

    def _build_mlp1(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, use_split_bn=False, num_splits=8):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            
            if l < num_layers - 1:
                # Hidden layers
                if use_split_bn:
                    mlp.append(SplitBatchNorm1d(dim2, num_splits=num_splits))
                else:
                    mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # Output layer
                # Follow SimCLR's design with optional split batch norm
                if use_split_bn:
                    mlp.append(SplitBatchNorm1d(dim2, affine=False, num_splits=num_splits))
                else:
                    mlp.append(nn.BatchNorm1d(dim2, affine=False))
        
        return nn.Sequential(*mlp)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True,num_splits=8):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                #mlp.append(nn.Linear(dim1, dim2, bias=False))
                #mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(SplitBatchNorm1d(dim2, num_splits=num_splits))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_param(self,moco_momentum):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * moco_momentum + param_q.data * (1. - moco_momentum)
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
       
    def forward_withoutpred_sym(self,im_q,im_k,moco_momentum):
        q = self.encoder_q(im_q, use_feature=False) # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        q_pred = q
        k_pred = self.encoder_q(im_k, use_feature=False)  # queries: NxC
        k_pred = nn.functional.normalize(k_pred, dim=1)
        with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
            
            self._momentum_update_key_encoder_param(moco_momentum)# update the key encoder
           
            im_q_, idx_unshuffle = self._batch_shuffle_single_gpu(im_q)
            q = self.encoder_k(im_q_, use_feature=False)  # keys: NxC
            q = nn.functional.normalize(q, dim=1)
            q = self._batch_unshuffle_single_gpu(q, idx_unshuffle)
            q = q.detach()

            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_, use_feature=False)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
            k = k.detach()
        return q_pred, k_pred, q, k
    def forward_withoutpred_multicrop(self,im_q_list,im_k,moco_momentum):
        q_list = []
        for im_q in [im_k]+im_q_list:
            q = self.encoder_q(im_q, use_feature=False)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            q_list.append(q)
        key_list = []
        with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
            self._momentum_update_key_encoder_param(moco_momentum)# update the key encoder
            for key_image in [im_k]+im_q_list[:1]:
                key_image, idx_unshuffle1 = self._batch_shuffle_single_gpu(key_image)
                q = self.encoder_k(key_image, use_feature=False)  # keys: NxC
                q = nn.functional.normalize(q, dim=1)
                q = self._batch_unshuffle_single_gpu(q, idx_unshuffle1)
                q = q.detach()
                key_list.append(q)
        return q_list,key_list

    def forward(self, im_q, im_k,run_type=0,moco_momentum=0.999):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            type: 0:sym; 1:multicrop
        Output:
            logits, targets

        """
        if run_type==0:
            return self.forward_withoutpred_sym(im_q,im_k,moco_momentum)
        elif run_type==1:
            return self.forward_withoutpred_multicrop(im_q, im_k, moco_momentum)



class CaCo_PN(nn.Module):
    def __init__(self,bank_size,dim):
        super(CaCo_PN, self).__init__()
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self,q):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        logit=torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
