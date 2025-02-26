from federatedFrameW.base.flocalbase import lbase
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer
from federatedFrameW.utils.torch_utils import torch_to_numpy,numpy_to_torch,cosine_similarity
from torch.utils.data import DataLoader
from collections import Counter
import torch
import copy
import numpy as np
from collections import deque

MAX_LOSS = 9999

class lListDec(lbase):
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
            - local_mom: local momentum
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_samples = len(kwargs['valid_data'])
        self.validloader_full = DataLoader(kwargs['valid_data'], self.valid_samples)
        self.local_vls = [] # current valid loss of global models
        self.min_tls = []
        self.cos_bounds = []
        self.loss_bounds = []
        self.update = None
        print('Local %s: train_samples: %d, valid_samples: %d, test_samples: %d' % (self.id, self.train_samples, self.valid_samples, self.test_samples))
        # if self.id == '1':
        #     print(f'local{self.id} len(velocities): {len(self.optimizer.velocities)}')
        #     for v in self.optimizer.velocities: print(f'v.shape: {v.shape}')
            # FEMNIST LR: local1 len(velocities): 2 v.shape: torch.Size([62, 784]) v.shape: torch.Size([62])

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
        if self.optimizer_name == 'CustomSGD':
            self.temp_opt = get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate, momentum=self.mu)
            return get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate, momentum=self.mu)
        else:
            return get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate)

    def train(self, byzantine=False, toNumpy=False):
        ini_para = [p.data.clone() for p in self.model.parameters()]
        train_loss = 0
        # print(f'local trian, ini_para: {ini_para[0].data[0][:20]}')
        for lround in range(self.local_epochs): # a local_epoch is a batch, actually
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            if byzantine: self.grad_attack()
            # grad = [p.grad.data for p in self.model.parameters() if p.requires_grad]
            # print(f'local train, local{self.id}.grad: {grad[0].data[0][20:25]}')
            self.optimizer.step()
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, MAX_LOSS), loss)
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, MAX_LOSS), loss)
            train_loss += loss.cpu().detach().numpy()
        # self.grad = torch_to_numpy(grad)
        self.update = [a.data.clone()-b for a,b in zip(self.model.parameters(), ini_para)]
        self.local_tl = train_loss / self.local_epochs
        if toNumpy: 
            self.update = torch_to_numpy(self.update)
        # print(f'local trian, fnl_para: {list(self.model.parameters())[0].data[0][:20]}')
        # print(f'local train, local{self.id}.update: {self.update[0].data[0][:20]}') # not equal to grad*lr --> data precision differ

    def byzantine_train(self):
        '''simulating honest locals in empire/little/omniscient attack'''
        ini_para = [p.data.clone() for p in self.model.parameters()]
        self.model.train()
        for lround in range(self.local_epochs): # a local_epoch is a batch, actually
            X, y = self.get_next_temp_train_batch()
            self.temp_opt.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.temp_opt.step()
        self.temp_up = [a.data.clone()-b for a,b in zip(self.model.parameters(), ini_para)]

    def gen_corp_update(self, honest_ups,num_corp_locals=0):
        # print(f'len(honest_ups): {len(honest_ups)}, honest_ups[0].shape: {honest_ups[0].shape}, honest_ups[1]: {honest_ups[1].shape} mean_hup.shape: {mean_hup.shape} ')
        # femnist LR: len(honest_ups): 14, honest_ups[0].shape: (48670,), honest_ups[1]: (48670,) mean_hup.shape: (48670,) std_hup.shape: (48670,)
        mean_hup = np.average(honest_ups, axis=0)
        # print(f'mean_hup[10:15]: {mean_hup[10:15]}')

        if self.at == 'empire':
            self.update = mean_hup * (-self.empire_para) * (len(honest_ups) / num_corp_locals)
            # print(f'after empire, corp_up: {self.update[10:15]}')

        elif self.at == 'little':
            if self.little_z is None:
                n = len(self.sampled_locals)
                m = len(self.sampled_corp_locals)
                s = int(n / 2 + 1) - m
                p = (n - m - s) / (n - m)
                self.little_z = norm.ppf(p)
                print("little attack, z: %d, n: %d, m: %d, s: %d, p: %d"%(self.little_z,n,m,s,p)) 
            std_hup = np.std(honest_ups, axis=0)
            # print(f'std_hup[10:15]: {std_hup[10:15]}')
            self.update = mean_hup - self.little_z * std_hup
            # print(f'after little, corp_up: {self.update[10:15]}')

        elif self.at == 'omniscient':
            self.update = -mean_hup
            # print(f'after omniscient, corp_up: {self.update[10:15]}')
        
        else:
            raise RuntimeError("local gen_corp_update: Undefined Byzantine attack!")


    def grad_attack(self):
        assert(self.at in ['clean', 'signflip', 'gauss', 'labelflip'])
        if self.at is None or self.at == 'clean' or self.at == 'labelflip':
            return

        if self.at == 'signflip':
            for p in self.model.parameters():
                if p.requires_grad: p.grad.data = -p.grad.data
        
        elif self.at == 'gauss':
            for p in self.model.parameters():
                if p.requires_grad: p.grad.data = torch.randn(*p.grad.shape).to(p.grad.device).mul(p.grad.std())

        # elif self.at == 'empire':
        #     for p in self.model.parameters():
        #         if p.requires_grad: p.grad.data = p.grad.data.mul(-self.empire_para)
        

    def test_mp(self):
        '''test multiple local models in local.para_groups, return the highest accumulated test_acc, 
        and corresponding test_loss, tested by the whole local dataset.'''
        test_accs = []
        test_losses = []

        for para in self.para_groups: 
            # numpy_to_torch(para, self.model) # set lmodel params
            self.set_model_parameters(para)
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
            test_accs.append(test_acc)
            test_loss = torch.where(torch.isinf(test_loss), torch.full_like(test_loss, MAX_LOSS), test_loss)
            test_loss = torch.where(torch.isnan(test_loss), torch.full_like(test_loss, MAX_LOSS), test_loss)
            test_losses.append(test_loss.cpu().detach().numpy())

        return test_accs, test_losses, self.test_samples
    
    def vote_mp(self,gupdate=None):
        '''voting phase, test multiple models and vote, tested by the whole local dataset'''
        # if self.id == '1': print(f'before voting, local1 min_tls: {self.min_tls}')
        for para in self.para_groups:
            # numpy_to_torch(para, self.model) # set gmodel params
            self.set_model_parameters(para)
            self.model.eval()
            loss = 0
            for x, y in self.validloader_full:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss += self.loss(output, y)
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, MAX_LOSS), loss)
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, MAX_LOSS), loss)
            loss = loss.cpu().detach().numpy()
            self.local_vls.append(loss)
            if len(self.para_groups) > 1: # the first global round
                self.min_tls.append(min(self.local_vls[0], loss))
        
        if self.at == 'clean':
            self.local_vls = [loss]
        
        # update min valid loss, only in eps strategy
        if self.vloss_eps > 1:
            if len(self.para_groups) == 1:
                self.min_tls.append(min(self.min_tls[self.brad_gmdl_idx], self.local_vls[-1]))
            self.min_tls[self.brad_gmdl_idx] = self.min_tls[-1]

        # update the loss bound
        # self.loss_bounds[self.brad_gmdl_idx].append(self.local_vls[-1] - self.local_vls[self.brad_gmdl_idx])
        # self.loss_bounds.append(copy.deepcopy(self.loss_bounds[self.brad_gmdl_idx])) # the new model's loss_queue is the same as its parent model
        # if self.id == '1': self.print_bounds(self.loss_bounds)

        # update local cos bound
        if self.queue_size > 0:
            if self.local_vls[-1] < self.local_vls[self.brad_gmdl_idx]:
                if self.update is None: self.train()
                self.cos_bounds[self.brad_gmdl_idx].append(cosine_similarity(torch_to_numpy(gupdate), torch_to_numpy(self.update)))
                # if self.id == '1': self.print_bounds(self.cos_bounds)
            self.cos_bounds.append(copy.deepcopy(self.cos_bounds[self.brad_gmdl_idx]))
            # if self.id == '1': print(f'after voting, local1 min_tls: {self.min_tls}')
        
        return 0, 1, 1, self.local_vls

    def print_bounds(self,bounds=[]):
        qstr='MDL0: '
        for i,q in enumerate(bounds):
            print(f'MDL{i}: {q}')
        #     qstr.join(map(str,q))
        #     qstr += '\tMDL'+str(i+1)+': '
        # print(qstr)
