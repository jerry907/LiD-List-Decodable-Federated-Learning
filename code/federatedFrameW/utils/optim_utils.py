from torch.optim import Optimizer, SGD, Adam, Adagrad
from abc import ABC
import numpy as np
import torch
import copy

def get_optimizer(optim_name):
    Optim_List = {
        'SGD': SGD
        , 'SGD_lrs': SGD_lrs
        , 'CustomSGD': CustomSGD
        , 'Adam': Adam
        , 'Adagrad': Adagrad
        , 'ns_Optimizer': ns_Optimizer
        , 'kl_Optimizer': kl_Optimizer
        , 'ns_iter_Optimizer': ns_iter_Optimizer
    }
    return Optim_List[optim_name]


class SGD_lrs(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['learning_rate']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGD_lrs, self).__init__(params, defaults)

    def step(self, eta=None):
        res_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if eta:
                        p.data = p.data - eta * p.grad.data
                    else:
                        p.data = p.data - group['lr'] * p.grad.data
                    res_grad.append(p.grad.data)

        return group['params'], res_grad


class ns_iter_Optimizer(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['personal_learning_rate']
        lamda = hyperparams['lamda']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr
                        , lamda=lamda
                        )
        super(ns_iter_Optimizer, self).__init__(params, defaults)

    def step(self, s_model):
        s_update = s_model.copy()
        for group in self.param_groups:
            for p, s_param in zip(group['params'], s_update):
                if p.requires_grad:
                    p.data = (p.data - group['lr'] * p.grad.data + group['lr'] * group['lamda'] * s_param.data) / (
                            1 + group['lr'] * group['lamda'])
        return group['params']


class ns_Optimizer(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['personal_learning_rate']
        lamda = hyperparams['lamda']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr
                        , lamda=lamda
                        )
        super(ns_Optimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated):
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                if p.requires_grad:
                    p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data)
                    )
        return group['params']


class kl_Optimizer(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['personal_learning_rate']
        lamda = hyperparams['lamda']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr
                        , lamda=lamda
                        )
        super(kl_Optimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated):
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                if p.requires_grad:
                    p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data)
                    )
        return group['params']

class CustomSGD(Optimizer):  
    def __init__(self, params, lr=0.01, momentum=0.9, velocities=None):  
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)

        self.lr = lr
        self.momentum = momentum
        self.set_velocities(velocities)
        # print(f'optimizer.lr: {self.lr}, mom: {self.momentum}, len(param_groups): {len(self.param_groups)}')

    def set_velocities(self, velocities=None):
        if velocities is None:
            self.velocities = [torch.zeros_like(p.data) for p in self.param_groups[0]['params'] if p.requires_grad]
            # for group in self.param_groups:
            #     for param in group['params']:  
            #         if param.requires_grad: self.velocities.append(torch.zeros_like(p.data))
        else:
            self.velocities = copy.deepcopy(velocities)
        # print(f'set localv as: {self.velocities[0].data[0][20:25]}')
  
    def step(self):
        # print(f'optimizer len(param_groups): {len(self.param_groups)}')
        for group in self.param_groups:
            # print(f'velocities1: {self.velocities[0].data[0][:20]}')
            # print('local train, param1: ', self.param_groups[0]['params'][0].data[0][:20])
            for param, velocity in zip(group['params'], self.velocities):  
                if param.requires_grad:
                    velocity.data.mul_(self.momentum).add_(param.grad.data)
                    param.data.add_(velocity.data.mul(-self.lr))
            # print(f'local train, localv: {self.velocities[0].data[0][20:25]}')
            # print(f'local train, localv.mul(-lr): {self.velocities[0].data.mul(-self.lr).data[0][:20]}')
            # print('local train, param2: ', self.param_groups[0]['params'][0].data[0][:20])
