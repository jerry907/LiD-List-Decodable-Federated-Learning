from federatedFrameW.base.fglobalbase import gbase
from federatedFrameW.utils.torch_utils import torch_to_numpy,numpy_to_torch
from federatedFrameW.utils.aggr_utils import aggrs
import numpy as np
import math
import copy
import torch
MAX_LOSS = 9999

class gSigMdl(gbase):
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
        # self.total_train_samples = kwargs['total_train_samples']
        # results recorder
        self.num_no_loss_drop = 0
        self.global_round = 0
        self.num_no_loss_drop_dic = {}
        # Single Model
        # self.gmdl = [p.data.clone() for p in self.model.parameters()] # tensor
        self.gmdl = torch_to_numpy(self.model.parameters()) # numpy
        self.gmom = None # for clean

    def sample_locals(self):
        num_aggregate_locals = min(self.num_aggregate_locals, len(self.candidate_locals))
        self.sampled_locals = np.random.choice(self.candidate_locals, num_aggregate_locals, replace=False)

    def sample_corp_locals(self):
        '''sample corrupted locals uniformly'''
        self.sampled_corp_locals = self.select_locals_uniform(num_aggregate_locals=math.floor(self.corp_rate*len(self.candidate_locals)))
        print(f'sampled_corp_locals: {[l.id for l in self.sampled_corp_locals]}')
        
    def aggregate_parameters(self, beta=1):
        '''update gmodel pool: aggregate sampled_locals then vote '''
        assert (self.candidate_locals is not None and len(self.candidate_locals) > 0)
        self.global_round += 1
        
        # aggregation
        if self.at in ['empire', 'little', 'omniscient']: 
            self.gen_corp_update()
            updates = [torch_to_numpy(l.update) for l in self.sampled_locals if l in self.hlocals]
            updates.extend([l.update for l in self.sampled_locals if l in self.sampled_corp_locals])
        else:
            updates = [torch_to_numpy(l.update) for l in self.sampled_locals]

        weights = [1] * len(self.sampled_locals)
        kwargs = {'in_eta': 1-self.corp_rate, 'out_eta': self.corp_rate, 'GM_iter': self.gm_iter, 'norm': self.norm_bound,}
        aggr = aggrs.get("ra_{}".format(self.ra))
        if aggr is None:
            raise NotImplementedError("Aggregator '{}' not implemented".format(aggr))
        ra_update = aggr(updates, weights=weights, **kwargs)
        # print(f'after {self.at} attack, ra_update: {ra_update[15:20]}')
        # if self.at == 'clean': # clean with single Gmodel, with random sampling as aggregator
        #     moms = [l.optimizer.velocities for l in self.sampled_locals]
        #     self.gmom = aggr(moms, weights, **kwargs)

        # self.gmdl = [a + b for a,b in zip(self.gmdl, ra_update)]
        self.gmdl += ra_update
        hlocal_tls = [l.local_loss for l in self.sampled_locals if l in self.hlocals]
        self.rs_train_loss.append(np.mean(hlocal_tls, axis=0))
    
    def gen_corp_update(self):
        honest_ups = [torch_to_numpy(l.update) for l in self.sampled_locals if l in self.hlocals]
        mean_hup = np.average(honest_ups, axis=0)
        # print(f'true mean_hup[10:15]: {mean_hup[10:15]}')

        if self.at == 'empire':
            corp_update = mean_hup * (-self.empire_para) * (len(self.hlocals) / len(self.sampled_corp_locals))
            # print(f'empire corp_up: {corp_update[10:15]}')

        elif self.at == 'little':
            if self.little_z is None:
                n = len(self.sampled_locals)
                m = len(self.sampled_corp_locals)
                s = int(n / 2 + 1) - m
                p = (n - m - s) / (n - m)
                self.little_z = norm.ppf(p)
                print("little attack, z: %d, n: %d, m: %d, s: %d, p: %d"%(self.little_z,n,m,s,p)) 
            std_hup = np.std(honest_ups, axis=0)
            corp_update = mean_hup - self.little_z * std_hup

        elif self.at == 'omniscient': # fedavg = - mean
            num_sampled_honest = len(honest_ups)
            num_sampled_corp = len(self.sampled_locals) - num_sampled_honest
            # print(f'before omn attack, mean: {mean_hup[15:20]} num_sampled_corp {num_sampled_corp}')
            corp_update = (-1) * mean_hup * (len(self.sampled_locals) + num_sampled_honest) / num_sampled_corp
            
        else:
            raise RuntimeError("local gen_corp_update: Undefined Byzantine attack!")
        
        for l in self.sampled_corp_locals: l.update = corp_update

    def send_para_groups(self):
        '''send gmdl to locals to evaluate'''
        numpy_to_torch(self.gmdl, self.model) # set global model
        for local in self.candidate_locals:
            # local.set_model_parameters(self.gmdl)
            local.set_model_parameters(list(self.model.parameters()))

    def evaluate(self):
        '''evaluate global model by candidate locals'''
        if (len(self.rs_train_loss) == 1):
            loss_drop = 0
        else:
            loss_drop = self.rs_train_loss[-1] - self.rs_train_loss[-2]
        
        if abs(loss_drop) >= 0.0001:
            self.num_no_loss_drop_dic[self.global_round] = self.num_no_loss_drop
            self.num_no_loss_drop = 0
        else:
            self.num_no_loss_drop += 1

        if (self.num_no_loss_drop > self.patience_iter):#  or self.rs_train_loss[-1] == MAX_LOSS:
            print("-------------Round number: {0:05d}".format(self.global_round), " -------------")
            print('No train loss drop made in {} iterations or MAX LOSS. Quitting.'.format(self.num_no_loss_drop))
            self.global_round = -1

        # global test
        glob_acc = 0
        if self.global_round == 1 \
        or self.global_round == 2 \
        or self.global_round == 5 \
        or (self.global_round % self.eval_every == 0) \
        or (self.global_round == -1):
            self.send_para_groups()
            stats = self.test()
            glob_acc = np.sum(stats[2], axis=0) / np.sum(stats[1])  # test acc pre sample
            test_loss = np.mean(stats[3], axis=0)  # t.detach().cpu().numpy() for t in
            self.rs_glob_acc.append(glob_acc)
            self.rs_test_loss.append(test_loss)
            print(self.name + "-" + "Global Testing Accurancy: %s" % (glob_acc))
            print(self.name + "-" + "Global Testing Loss: %s" % (test_loss))
            print(self.name + "-" + "AVG Train Loss of honest locals: %.4f" % (self.rs_train_loss[-1]))
        
        return glob_acc, self.global_round

    def test(self):
        '''evaluate multiple gmodels, report the best TA, remain local model unchanged'''
        tot_correct = []
        test_loss = []
        num_samples = []
        
        for c in self.candidate_locals:
            acc, tl, ns = c.test()
            tot_correct.append(acc) # total acc of all test samples on local c
            test_loss.append(tl)
            num_samples.append(ns)
        ids = [c.id for c in self.candidate_locals]

        return ids, num_samples, tot_correct, test_loss

            
