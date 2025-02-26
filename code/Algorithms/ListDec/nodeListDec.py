import random
import copy
from federatedFrameW.fnode.nodeCentralizedFL import nCentralizedFL
from federatedFrameW.utils.torch_utils import torch_to_numpy,numpy_to_torch
from collections import deque
from torch.utils.data import DataLoader
from federatedFrameW.utils.data_utils import tData_lf

MAX_LOSS = 9999


class nListDec(nCentralizedFL):
    '''
    List Decodable node.

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparameters: hyperparameters
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for fg in self.fglobals:
            fg.sample_corp_locals() # generate corp locals for gobal node
            fg.hlocals = [l for l in fg.candidate_locals if l not in fg.sampled_corp_locals]
            print('nListDec init. len(fg.candidate_locals): ', len(fg.candidate_locals))
            print('nListDec init. len(fg.sampled_corp_locals): ', len(fg.sampled_corp_locals))
            print('nListDec init. len(fg.hlocals): ', len(fg.hlocals))
            if self.queue_size > 0:
                for l in fg.candidate_locals:
                    # l.min_tls = [MAX_LOSS] * fg.num_listDec # historical min loss of global models
                    for i in range(fg.num_listDec):
                        l.loss_bounds.append(deque(maxlen=self.queue_size))
                        l.cos_bounds.append(deque(maxlen=self.queue_size))
            if self.at == 'labelflip':
                print('Flipping label--------')
                for l in fg.sampled_corp_locals: 
                    # l.trainloader_full = tData_lf(train_data=l.trainloader_full, dataset=self.dataset) # LF1
                    l.trainloader = tData_lf(train_data=l.trainloader, dataset=self.dataset, bs = self.batch_size) # LF2
                    l.iter_trainloader = iter(l.trainloader)
                    # for x, y in l.trainloader: 
                    #     print(f'local{l.id} after LF, train y[:10]: {y[:10]}')
                    #     break
            elif self.at in ['empire', 'little', 'omniscient']:
                print(f'{self.at} prepare temp_trainloader!')
                for l in fg.hlocals:
                    newx, newy = [], []
                    for x, y in l.trainloader_full: 
                        newx.extend(x)
                        newy.extend(y)
                    l.temp_trainloader = DataLoader([(xi, yi) for xi, yi in zip(newx, newy)], self.batch_size)
                    l.temp_iter_train = iter(l.temp_trainloader)

    def train(self, sampled_only=False, round=0):
        for fg in self.fglobals:
            idx = random.randint(0, len(fg.para_groups)-1)
            fg.crt_mdl = fg.para_groups[idx]
            fg.brad_gmdl.append(idx)
            fg.paras_t[idx] += 1
            for fl in fg.candidate_locals:
                fl.set_model_parameters(fg.crt_mdl) # send global model to all locals
                fl.optimizer.set_velocities(fg.moms[idx])

            if self.sampled_only == 'true':
                if len(fg.sampled_locals) == 1: # sample one client for local training
                    fl = fg.sampled_locals[0]
                    honest_ups,honest_moms = [],[]
                    if fl in fg.sampled_corp_locals: # sampled a Byzantine local
                        if self.at in ['empire', 'little', 'omniscient']:
                            for hl in fg.hlocals: 
                                hl.byzantine_train()
                                honest_ups.append(torch_to_numpy(hl.temp_up))
                                # honest_moms.append(torch_to_numpy(hl.temp_opt.velocities)) # Byzantine clients return local momentum to server
                            fl.gen_corp_update(honest_ups,num_corp_locals=len(fg.sampled_corp_locals))
                        else:
                            fl.train(byzantine=True)
                    else: fl.train() # sampled an honest local
                elif len(fg.sampled_locals) == len(fg.candidate_locals):
                    for fl in fg.hlocals: fl.train(toNumpy=True)
                    if self.at in ['empire', 'little', 'omniscient']:
                        honest_ups = [hl.update for hl in fg.hlocals] 
                        for fl in fg.sampled_corp_locals: fl.gen_corp_update(honest_ups,num_corp_locals=len(fg.sampled_corp_locals))
                    else:
                        for fl in fg.sampled_corp_locals: fl.train(byzantine=True,toNumpy=True)
                    
            else:
                for fl in fg.candidate_locals:
                    if fl in fg.sampled_corp_locals: # sampled a Byzantine local
                        if self.at in ['empire', 'little', 'omniscient']:
                            if len(honest_ups) == 0:
                                for hl in fg.hlocals: 
                                    hl.byzantine_train()
                                    honest_ups.append(torch_to_numpy(hl.temp_up))
                            fl.gen_corp_update(honest_ups)
                        else:
                            fl.train(byzantine=True)
                    else: fl.train() # sampled an honest local

    def OLDtrain(self, sampled_only=False, round=0):
        if self.sampled_only == 'true':
            for fg in self.fglobals:
                idx = random.randint(0, len(fg.para_groups)-1)
                fg.crt_mdl = fg.para_groups[idx]
                fg.brad_gmdl.append(idx)
                fg.paras_t[idx] += 1
                # numpy_to_torch(sample_p, fg.model) # set global model
                for fl in fg.sampled_locals:
                    # fl.set_local_parameters(list(fl.model.parameters())) # store current local model
                    fl.set_model_parameters(fg.crt_mdl) # send global model to local
                    fl.optimizer.set_velocities(fg.moms[idx])
                    if fl in fg.sampled_corp_locals: fl.byzantine_train()
                    else: fl.train()
        else:
            for fg in self.fglobals:
                # randomly sample a param from global param pool
                idx = random.randint(0, len(fg.para_groups)-1)
                fg.crt_mdl = fg.para_groups[idx]
                fg.brad_gmdl.append(idx)
                fg.paras_t[idx] += 1
                # numpy_to_torch(sample_p, fg.model) # set global model
                for fl in fg.candidate_locals:
                    # fl.set_local_parameters(list(fl.model.parameters())) # store current local model
                    fl.set_model_parameters(fg.crt_mdl)
                    fl.optimizer.set_velocities(fg.moms[idx])
                    if fl in fg.sampled_corp_locals: fl.byzantine_train()
                    else: fl.train()
                    
    def eval_global(self):
        for fg in self.fglobals:
            global_round = fg.evaluate_mp()[1] # res.append([glob_acc, train_acc, train_loss])
        return global_round