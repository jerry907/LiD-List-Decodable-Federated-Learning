from federatedFrameW.base.flocalbase import lbase
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer
from federatedFrameW.utils.torch_utils import torch_to_numpy,numpy_to_torch
from torch.utils.data import DataLoader
from collections import Counter
import torch
import copy
import numpy as np

MAX_LOSS = 9999


class lSigMdl(lbase):
    '''
    local Class for List Decodable

    kwargs:
        - id: local id
        - device: device
        - model: calculation model deepcopy
        - hyperparams:
            - batch_size: batch size
            - local_epochs: int, number of epochs for local training
            - optimizer_name: str, name of optimizer
            - loss: loss function
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_samples = len(kwargs['valid_data'])
        if self.valid_samples > 0:
            self.validloader_full = DataLoader(kwargs['valid_data'], self.valid_samples)
        self.local_tls = [] # current valid loss of global models
        print('Local %s: train_samples: %d, valid_samples: %d, test_samples: %d' % (self.id, self.train_samples, self.valid_samples, self.test_samples))

    def gen_loss(self):
        if self.loss_name == 'mcFocalLoss':
            # number of samples per class in the training dataset
            if self.dataset == 'femnist':
                samples_per_class = np.zeros(62)
            elif self.dataset == 'cifar':
                samples_per_class = np.zeros(10)

            train_y = [y.numpy() for x,y in self.trainloader_full]
            for key,c in Counter(train_y[0]).items():
                samples_per_class[key] = c
            
            alpha = sum(samples_per_class) / (samples_per_class + 1e-5)
            alpha = alpha / sum(alpha)  # normalization, prevent loss inf
            print('alpha: ', alpha)
            return get_loss(self.loss_name)(gamma=self.gamma, alpha=alpha.tolist())

        elif self.loss_name == 'balancedFocalLoss':
            # number of samples per class in the training dataset
            if self.dataset == 'femnist':
                samples_per_class = np.zeros(62)
            elif self.dataset == 'cifar':
                samples_per_class = np.zeros(10)

            train_y = [y.numpy() for x,y in self.trainloader_full]
            for key,c in Counter(train_y[0]).items():
                samples_per_class[key] = c

            return get_loss(self.loss_name)(
                        loss_type="focal_loss",
                        beta=self.fl_beta,
                        fl_gamma=self.gamma,
                        samples_per_class=samples_per_class + 1e-5, # prevent devided by 0
                        class_balanced=True
                    )
        else:
            return get_loss(self.loss_name)()

    def gen_optimizer(self):
        return get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate, momentum=self.mu)

    def train(self,round=0):
        # print(f'self.local_epochs: {self.local_epochs}')
        ini_para = [p.data.clone() for p in self.model.parameters()]
        local_losses = []
        self.model.train()
        # print(f'local trian, ini_para: {ini_para[0].data[0][:20]}')
        for r in range(self.local_epochs): # a local_epoch is a batch, actually
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            # grad = [p.grad.data for p in self.model.parameters() if p.requires_grad]
            # print(f'local train, local{self.id}.grad: {grad[0].data[0][20:25]}')
            self.optimizer.step()
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, MAX_LOSS), loss)
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, MAX_LOSS), loss)
            local_losses.append(loss.cpu().detach().numpy())
        # self.grad = torch_to_numpy(grad)
        self.update = [a.data.clone()-b for a,b in zip(self.model.parameters(), ini_para)]
        self.local_loss = np.mean(local_losses)
        # print(f'local trian, fnl_para: {list(self.model.parameters())[0].data[0][:20]}')
        # print(f'local train, local{self.id}.update: {self.update[0].data[0][:20]}') # not equal to grad*lr --> data precision differ

    def byzantine_train(self,round=0):
        if self.at in ['empire', 'little', 'omniscient']:
            return
            
        ini_para = [p.data.clone() for p in self.model.parameters()]
        local_attack_losses = []
        self.model.train()
        # print(f'local trian, ini_para: {ini_para[0].data[0][:20]}')
        for local_batch in range(self.local_epochs): # a local_epoch is a batch, actually
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.grad_attack()
            self.optimizer.step()
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, MAX_LOSS), loss)
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, MAX_LOSS), loss)
            local_attack_losses.append(loss.cpu().detach().numpy())
        self.update = [a.data.clone()-b for a,b in zip(self.model.parameters(), ini_para)]
        # self.local_loss = np.mean(local_attack_losses) # record train loss of honest clients only

    def grad_attack(self):
        if self.at is None or self.at == 'clean' or self.at == 'labelflip':
            return

        if self.at == 'signflip':
            for p in self.model.parameters():
                if p.requires_grad: p.grad.data = -p.grad.data
        
        elif self.at == 'gauss':
            for p in self.model.parameters():
                if p.requires_grad: p.grad.data = torch.randn(*p.grad.shape).to(p.grad.device).mul(p.grad.std())
        
        # else:
        #     raise RuntimeError("Undefined Byzantine attack!")

    def update_attack(self):
        if self.at is None or self.at == 'clean':
            return

        if self.at == 'signflip':
            for p in self.update:
                p.data = -p.data
        
        elif self.at == 'gauss':
            for p in self.update:
                p.data = torch.randn(*p.shape).to(p.device).mul(p.std())

        elif self.at == 'empire':
            for p in self.update:
                p.data = p.data.mul(-self.empire_para)
        
        else:
            raise RuntimeError("Undefined Byzantine attack!")

    def test(self):
        test_acc = 0
        test_loss = 0
        self.model.eval()
        for x, y in self.testloader_full:  # all test_data as one batch
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            if len(y.shape) > 1 and y.shape[1] > 1:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1))).item()
            else:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_loss += self.loss(output, y)
        
        test_loss = torch.where(torch.isinf(test_loss), torch.full_like(test_loss, MAX_LOSS), test_loss)
        test_loss = torch.where(torch.isnan(test_loss), torch.full_like(test_loss, MAX_LOSS), test_loss)

        return test_acc, test_loss.cpu().detach().numpy(), self.test_samples