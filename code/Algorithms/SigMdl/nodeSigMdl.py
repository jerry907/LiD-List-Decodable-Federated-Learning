import random
import copy
from federatedFrameW.fnode.nodeCentralizedFL import nCentralizedFL
from federatedFrameW.utils.torch_utils import torch_to_numpy,numpy_to_torch
from federatedFrameW.utils.data_utils import tData_lf

MAX_LOSS = 9999


class nSigMdl(nCentralizedFL):
    '''
    Single Model node.

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
            print('nSigMdl init. len(fg.candidate_locals): ', len(fg.candidate_locals))
            print('nSigMdl init. len(fg.sampled_corp_locals): ', len(fg.sampled_corp_locals))
            print('nSigMdl init. len(fg.hlocals): ', len(fg.hlocals))
            if self.at == 'labelflip':
                print('LF corp data!')
                for l in fg.sampled_corp_locals: 
                    l.trainloader = tData_lf(train_data=l.trainloader, dataset=self.dataset, bs = self.batch_size) # LF2
                    l.iter_trainloader = iter(l.trainloader)

    def train(self, sampled_only=False, round=0):
        if self.sampled_only == 'true':
            for fg in self.fglobals:
                numpy_to_torch(fg.gmdl, fg.model) # set global model
                for fl in fg.sampled_locals:
                    fl.set_model_parameters(list(fg.model.parameters())) # send global model to local
                    # fl.optimizer.set_velocities(fg.gmom) # clean
                    if fl in fg.sampled_corp_locals: fl.byzantine_train()
                    else: fl.train()
        else:
            for fg in self.fglobals:
                numpy_to_torch(sample_p, fg.model) # set global model
                for fl in fg.candidate_locals:
                    fl.set_model_parameters(list(fg.model.parameters()))
                    # fl.optimizer.set_velocities(fg.gmom) # clean
                    if fl in fg.sampled_corp_locals: fl.byzantine_train()
                    else: fl.train()
                    
    def eval_global(self):
        for fg in self.fglobals:
            global_round = fg.evaluate()[1] # res.append([glob_acc, train_acc, train_loss])
        return global_round