from federatedFrameW.base.fglobalbase import gbase
from federatedFrameW.utils.torch_utils import torch_to_numpy,numpy_to_torch,cosine_similarity
from federatedFrameW.utils.aggr_utils import aggrs
import numpy as np
import math
import copy
import torch
MAX_LOSS = 9999
MAX_EPS = 1.25

class gListDec(gbase):
    '''
    global Class for List Decodable

    kwargs:
        - id: id of the global
        - model: calculation model deepcopy
        - hyperparame:
            - num_aggregate_locals: number of local models to aggregate
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # results recorder
        self.old_loss = MAX_LOSS
        self.num_no_loss_drop = 0
        self.global_round = 0
        self.para_group_update = []
        self.votes = []
        self.avg_hlocal_vls = []
        self.min_valid_loss = []
        self.brad_gmdl = [] # broadcast gmdl
        self.paras_ts=[]
        # self.num_no_loss_drop_dic = {}
        self.cor_tl = {} # train loss of good round 
        self.clean_tl = []
        self.vote_false_rate = {}
        self.vote_right_rate = {}
        self.loss_up = {}
        # List Decodable
        # self.num_listDec = math.ceil(1/(1-self.corp_rate)) # OLD LS
        self.num_listDec = int(1 // (1-self.corp_rate)) # NEW LS
        if self.list_size > 0: self.num_listDec = self.list_size
        if self.at == 'clean': self.num_listDec = 1
        self.para_groups = [[p.data.clone() for p in self.model.parameters()]] * self.num_listDec
        self.paras_t = [0] * self.num_listDec
        self.if_cor = [True] * self.num_listDec # the global model is correct or malicious
        self.moms = [None] * self.num_listDec
        print(f'num_listDec: {self.num_listDec}') #FEMNIST_LR: len(self.para_groups[0]): 2, len(self.para_groups[1]): 2
        # for p in self.para_groups[0]: print(f'{p.data.shape}') #FEMNIST_LR: torch.Size([62, 784]) torch.Size([62])

    def sample_locals(self):
        num_aggregate_locals = min(self.num_aggregate_locals, len(self.candidate_locals))
        if self.at == 'clean': #single model
            self.sampled_locals = np.random.choice(self.hlocals, 1, replace=False)
        else:
            self.sampled_locals = np.random.choice(self.candidate_locals, num_aggregate_locals, replace=False)
        # if self.sampled_locals[0].id not in self.corp_local_idx:
        #     print(f'{self.global_round+1}r Sampled local{self.sampled_locals[0].id} is honest.')

    def sample_corp_locals(self):
        '''sample corrupted locals uniformly'''
        self.sampled_corp_locals = self.select_locals_uniform(num_aggregate_locals=math.floor(self.corp_rate*len(self.candidate_locals)))
        self.num_hlocals = len(self.candidate_locals) - len(self.sampled_corp_locals)
        self.corp_local_idx = [l.id for l in self.sampled_corp_locals]
        print(f'sampled_corp_locals: {self.corp_local_idx}')
        
    def aggregate_parameters(self, beta=1):
        '''update gmodel pool: aggregate sampled_locals then vote '''
        assert (self.candidate_locals is not None and len(self.candidate_locals) > 0)
        self.global_round += 1
        if self.global_round % 2000 == 0:
            self.vloss_eps = min(self.vloss_eps+self.eps_cof, MAX_EPS)

        # aggregation
        # ra_moms = [l.optimizer.velocities for l in self.sampled_locals]
        # ra_updates = [l.update for l in self.sampled_locals]
        moms = [torch_to_numpy(l.optimizer.velocities) for l in self.sampled_locals]
        updates = [l.update for l in self.sampled_locals]
        weights = [1] * len(self.sampled_locals)
        kwargs = {'in_eta': 1-self.corp_rate, 'out_eta': self.corp_rate, 'GM_iter': self.gm_iter, 'norm': self.norm_bound, 'num_hls': self.num_hlocals}
        aggr = aggrs.get("ra_{}".format(self.ra))
        if aggr is None:
            raise NotImplementedError("Aggregator '{}' not implemented".format(self.ra))
        ra_update, ra_mom = aggr(updates, moms, weights, **kwargs)
        # print(f'len(ra_update) ra_mom: {len(ra_update), len(ra_mom)}')

        if isinstance(ra_update, np.ndarray):  # transfer ra_update to tensor
            # print('---transfer ra_update to tensor!')
            # print(f'---before transfer, len(ra_update): {len(ra_update)}') # 48670
            numpy_to_torch(ra_update, self.model)
            ra_update = [p.data.clone() for p in self.model.parameters()]
            # print(f'---after transfer, len(ra_update): {len(ra_update)}') # 2 torch.Size([62, 784]) torch.Size([62])
            # for u in ra_update: print(f'u.shape {u.shape}')

        if ra_mom is not None and isinstance(ra_mom, np.ndarray):
            # print('---transfer ra_mom to tensor!')
            # print(f'---before transfer, len(ra_mom): {len(ra_mom)}')
            numpy_to_torch(ra_mom, self.model)
            ra_mom = [p.data.clone() for p in self.model.parameters()]
            # print(f'---after transfer, len(ra_mom): {len(ra_mom)}')
            # for m in ra_mom: print(f'm.shape {m.shape}')
        # print(f'after aggr: update: {ra_update[0].data[0][20:25]}\nmom: {ra_mom[0].data[0][20:25]}')

        # update global model
        idx_brad_mdl = self.brad_gmdl[-1]
        train_loss = None
        if self.at == 'clean': # single gmdl
            self.moms[idx_brad_mdl] = ra_mom
            self.para_groups = [[a+b for a,b in zip(self.crt_mdl, ra_update)]]
            self.vote_at = "worst"
            self.clean_tl.append(self.sampled_locals[0].local_tl)
        else: # multiple gmdls
            self.moms.append(ra_mom)
            self.para_groups.append([a+b for a,b in zip(self.crt_mdl, ra_update)])
            self.paras_t.append(0)
            if (self.if_cor[idx_brad_mdl] is True) and (self.sampled_locals[0].id not in self.corp_local_idx):
                self.if_cor.append(True)
                train_loss = self.sampled_locals[0].local_tl
            else: self.if_cor.append(False)

        self.paras_ts.append(copy.deepcopy(self.paras_t))
        if train_loss is not None: self.cor_tl[self.global_round] = train_loss
        # print(f'{self.global_round}r Gmdl if_cor: {self.if_cor}')
        self.vote_phase()

    def vote_phase(self):
        voter={}
        for idx in range(len(self.para_groups)):
            voter[idx] = 0
        idx_brad_mdl = self.brad_gmdl[-1]
        hlocal_vls = [] # honest local valid losses
        self.send_para_groups() # send global mdls to locals
        
        # local voting
        for local in self.candidate_locals:
            _, _, _, local_vl = local.vote_mp(gupdate=self.sampled_locals[-1].update)
            if local in self.sampled_corp_locals:
                local_vote = self.byzantine_vote(local_vl)
            else:
                local_vote = np.argmin(local_vl)
                
                # vote scheme1: Dfac(loss+update)
                if self.queue_size > 0:
                    if local_vote == idx_brad_mdl and local_vl[-1] < local.min_tls[idx_brad_mdl] * self.vloss_eps: 
                        if local.update is None: local.train() # local model?
                        cos_sim = cosine_similarity(torch_to_numpy(self.sampled_locals[-1].update), torch_to_numpy(local.update))
                        if cos_sim > 0 and (len(local.cos_bounds[idx_brad_mdl]) == 0 or cos_sim < np.max(local.cos_bounds[idx_brad_mdl])):
                            local_vote = len(local_vl)-1
                # vote scheme2: eps
                elif self.vloss_eps > 1 and local_vl[-1] < local.min_tls[idx_brad_mdl] * self.vloss_eps:
                    temp1 = local_vl[idx_brad_mdl] 
                    local_vl[idx_brad_mdl] = MAX_LOSS
                    local_vote = np.argmin(local_vl)
                    local_vl[idx_brad_mdl] = temp1
                   
                hlocal_vls.append(local_vl)
            voter[local_vote] += 1

        if self.if_cor[-1] is True: # good round
            self.vote_right_rate[self.global_round] = voter[len(self.para_groups)-1] > voter[idx_brad_mdl]
        elif self.if_cor[idx_brad_mdl] is True: # and self.sampled_locals[-1] in self.sampled_corp_locals
            self.vote_false_rate[self.global_round] = voter[len(self.para_groups)-1] > voter[idx_brad_mdl]

        havg_vl = np.mean(hlocal_vls, axis=0).tolist()
        self.avg_hlocal_vls.append(copy.deepcopy(havg_vl)) # record valid loss of global models, write into excel.
        self.votes.append(list(voter.values()))

        # decay of old gmdls' vote
        # for idx,val in voter.items():
        #     voter[idx]= val * (self.vote_decay ** self.paras_t[idx])

        # update global param pool
        min_vote = min(voter.values())  
        keys_with_min_vote = [k for k, v in voter.items() if v == min_vote]
        rm_key = keys_with_min_vote[-1]
        if len(keys_with_min_vote) > 1:
            temp = [self.paras_t[k] for k in keys_with_min_vote]
            rm_key = keys_with_min_vote[np.argmax(temp)]
        self.update_local_paras(rm_key)
        
        # record if newly aggregated para is voted
        if (len(self.para_groups)-1) != rm_key:
            self.para_group_update.append(self.global_round)
        
        # update global param pool and record best Gmdl Trainning Loss
        if self.at != 'clean':
            self.para_groups.pop(rm_key)
            self.moms.pop(rm_key)
            self.paras_t.pop(rm_key)
            havg_vl.pop(rm_key)
            self.if_cor.pop(rm_key)
            # self.if_cor[np.argmin(havg_vl)] = True

        if self.global_round > 1:
            self.old_loss = self.min_valid_loss[-1]
        self.min_valid_loss.append(min(havg_vl))
        if self.min_valid_loss[-1] > self.old_loss: # loss increase
            self.loss_up[self.global_round] = self.sampled_locals[-1] in self.hlocals

    def byzantine_vote(self, local_vl):
        if self.vote_at == "worst":
            local_vote = np.argmax(local_vl)
        elif self.vote_at == "random":
            print('Byzantine vote: random.')
            idxs = [i for i in range(len(local_vl))]
            idxs.remove(np.argmin(local_vl))
            local_vote = np.random.choice(idxs, 1, replace=False)[0]  
        else:
            local_vote = 0
            raise RuntimeError("Undefined byzantine_vote strategy!")
        return local_vote
        
    def send_para_groups(self, test=False):
        '''voting phase: send new gmdls to locals'''
        if (self.global_round == 1) or (self.at == 'clean') or (test is True):
            new_paras = self.para_groups
        else: 
            new_paras = [self.para_groups[-1]]
        for local in self.candidate_locals:
            local.para_groups = new_paras
            local.crt_mdl_cor = self.if_cor[-1] # if the new model is correct
            local.brad_gmdl_idx = self.brad_gmdl[-1]

    def evaluate_mp(self):
        '''evaluate global models in global.para_groups by candidate locals, report the highest TA of each local'''
        # check if stop training, self.patience_iter rounds train_loss did not decrease
        loss_drop = self.min_valid_loss[-1] - self.old_loss
        if abs(loss_drop) >= 0.0001:
            # self.num_no_loss_drop_dic[self.global_round] = self.num_no_loss_drop
            self.num_no_loss_drop = 0
        else:
            self.num_no_loss_drop += 1

        if (self.num_no_loss_drop > self.patience_iter):
            print("-------------Round number: {0:05d}".format(self.global_round), " -------------")
            print('No train loss drop made in {} iterations or MAX LOSS encountered. Quitting.'.format(self.num_no_loss_drop))
            self.global_round = -1

        # global test
        glob_acc = 0
        if self.global_round == 1 or self.global_round == 2 or self.global_round == 5 or ((self.global_round)%self.eval_every ==0) or (self.global_round == -1):
            self.send_para_groups(test=True)
            stats = self.test_mp()
            glob_acc = np.sum(stats[2], axis=0) / np.sum(stats[1])  # test acc pre sample
            test_loss = np.mean(stats[3], axis=0)
            self.rs_glob_acc.append(max(glob_acc))
            self.rs_test_loss.append(test_loss[np.argmax(glob_acc)])
            print(self.name + "-" + "Global Testing Accurancy: %s, max_acc: %.4f" % (glob_acc, self.rs_glob_acc[-1]))
            print(self.name + "-" + "Global Testing Loss: %s, recorded test loss: %.4f" % (test_loss, self.rs_test_loss[-1]))
            print(self.name + "-" + "best Gmdl valid Loss: %.4f" % (self.min_valid_loss[-1]))
        
        return glob_acc, self.global_round

    def test_mp(self):
        '''evaluate multiple gmodels, report the best TA, remain local model unchanged'''
        num_samples = []
        test_loss = []
        tot_correct = []
        for c in self.candidate_locals:
            accs, tls, ns = c.test_mp()
            tot_correct.append(accs) # total acc of all test samples on local c
            num_samples.append(ns)
            test_loss.append(tls)
        ids = [c.id for c in self.candidate_locals]

        return ids, num_samples, tot_correct, test_loss

    def update_local_paras(self, rm_key):
        '''update local para_groups, retain the voted gmdls only'''
        for c in self.candidate_locals:
            c.local_vls.pop(rm_key)
            c.update = None
            if self.vloss_eps > 1: c.min_tls.pop(rm_key)
            if self.queue_size > 0: c.cos_bounds.pop(rm_key)

    def print_server_ps(self):
        '''print global.para_groups'''
        print('len(self.para_groups): ', len(self.para_groups))
        for para in self.para_groups:
            print(para[:5])

            
