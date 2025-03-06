# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure=None):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer, no weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    p.mul_(1 - g['lr'] * g['weight_decay'])

                # Perform optimization step
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = g['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(dp, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(dp, dp, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(g['eps'])
                step_size = g['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

from typing import Any, Callable, Dict, Optional, overload


from torch.optim.optimizer import Optimizer


class LARS2(torch.optim.Optimizer):
    """
    LARS optimizer with explicit exclusion of SplitBatchNorm parameters and biases
    from weight decay and LARS adaptation.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, 
                        trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    def _is_splitbatchnorm(self, p):
        """Check if the parameter belongs to a SplitBatchNorm layer"""
        if hasattr(p, '_module_name'):
            module_name = p._module_name.lower()
            return 'splitbatchnorm' in module_name or 'splitbn' in module_name
        return False
    
    def _is_bias(self, p):
        """Check if the parameter is a bias"""
        if hasattr(p, '_param_name'):
            return 'bias' in p._param_name.lower()
        return False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                # Skip weight decay for SplitBatchNorm and bias parameters
                if weight_decay != 0 and not (self._is_splitbatchnorm(p) or self._is_bias(p)):
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # Apply LARS adaptation only for non-SplitBatchNorm and non-bias parameters
                if not (self._is_splitbatchnorm(p) or self._is_bias(p)):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(d_p)
                    
                    if param_norm != 0 and update_norm != 0:
                        # LARS coefficient
                        lars_coef = trust_coefficient * param_norm / update_norm
                        d_p = d_p.mul(lars_coef)
                
                # Apply momentum and update
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p)
                
                p.add_(buf, alpha=-lr)
        
        return loss

# Helper function to set parameter attributes for better tracking
def set_module_name_to_params(model):
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param._module_name = name
            param._param_name = param_name
            
# Example of how to set up parameter groups with your ResNet model
def setup_optimizer_with_no_lr_scheduler_for_projection_head(model, base_lr=0.1, 
                                                           weight_decay=1e-6, 
                                                           momentum=0.9,
                                                           trust_coefficient=0.001):
    # Tag parameters with their module names for better identification
    set_module_name_to_params(model)
    
    # Separate parameters into projection head and the rest of the model
    projection_head_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if name.startswith('fc.'):
            projection_head_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': projection_head_params, 'lr': base_lr, 'weight_decay': weight_decay}
    ]
    
    # Create the optimizer with these groups
    optimizer = LARS(param_groups, lr=base_lr, weight_decay=weight_decay, 
                     momentum=momentum, trust_coefficient=trust_coefficient)
    
    # Create your LR scheduler, but only apply it to the non-projection head parameters
    def lr_lambda(epoch, total_epochs=100):  # Default total_epochs=100
        # Define your learning rate schedule logic here
        # For example, a cosine decay schedule:
        return 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    
    # This scheduler will only adjust the learning rate for the first parameter group (non-projection head)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=[lambda epoch: lr_lambda(epoch), lambda _: 1.0]
    )
    
    return optimizer, scheduler

        
