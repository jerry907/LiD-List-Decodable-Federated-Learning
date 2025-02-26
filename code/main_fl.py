import argparse
import random
import numpy as np
import torch
import os
import sys
import time

from federatedFrameW.utils.model_utils import make_print_to_file
from Algorithms.ListDec.trainListDec import ListDec
from Algorithms.SigMdl.trainSigMdl import SigMdl


from federatedFrameW.models.models import Mclr_Logistic, DNN, CNNCifar2, Sent140_LSTM, Shkspr_LSTM, \
    Mclr_Logistic_Femnist, DNN_Femnist, CNN_Femnist, Mclr_CrossEntropy_Femnist


def main(*args, **kwargs):
    '''
    kwargs:
        - gpu: the device to train the model on
        - dataset: the name of the dataset
        - name: the name of the algorithm
        - model_name: the origin model for deepcopy
        - loss_name: the loss function to use
        - optimizer_name: the optimizer to use
        - batch_size: batch size
        - total_epochs: total number of epochs
        - local_epochs: int, number of epochs for local training
        - num_aggregate_locals: number of local models to aggregate
        - learning_rate: learning rate
        - times: the number of times to repeat the experiment
        - eta: the extra parameter, SGD_lrs optm?
        - local_mom: local momentum
    '''
    name =  kwargs['name']
    dataset = kwargs['dataset']
    model_name = kwargs['model_name']
    times = kwargs['times']
    gpu = kwargs['gpu']
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    print('device: %s'%(device))

    SEED = 2024 + times
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    log_file = kwargs['ra'] + "_" + kwargs['at'] + "_" + str(kwargs['corp_rate']) + "_" + kwargs['vote_at']
    if kwargs['local_epochs'] > 1 : log_file += "_mlt"+ str(kwargs['local_epochs']) + "_" + str(times)
    # elif dataset == 'cifar10': log_file += "_lr"+ str(kwargs['learning_rate']) + "_" + str(times)
    # log_file += "_ls" + str(kwargs['list_size']) + "_" + str(times)
    else: log_file += "_eps" + str(kwargs['vloss_eps']) + "_" + str(times)
    
    log_fp = os.path.join(kwargs['logf'], log_file) # log full path  
    f = open(log_fp+'.log','w+')
    sys.stdout = f

    hyperparams = {
        'model_name': model_name
        , 'dataset': kwargs['dataset']
        , 'loss_name': kwargs['loss_name']
        , 'name': name
        , 'total_epochs': kwargs['total_epochs']
        , 'local_data_distrb': kwargs['local_data_distrb']
        , 'valid_rate': kwargs['valid_rate']
        # , 'gamma': kwargs['gamma']
        # , 'fl_beta': kwargs['fl_beta']
        # general FL
        , 'learning_rate': kwargs['learning_rate']
        , 'num_aggregate_locals': kwargs['num_aggregate_locals']
        , 'batch_size': kwargs['batch_size']
        , 'beta': kwargs['beta']
        , 'local_epochs': kwargs['local_epochs']
        , 'optimizer_name': kwargs['optimizer_name']
        , 'times': times
        , 'log_fp': log_fp
        , 'patience_iter': kwargs['patience_iter']
        , 'eval_every': kwargs['eval_every']
        # Byzantine
        , 'at': kwargs['at']
        , 'empire_para': kwargs['empire_para']
        , 'little_z': kwargs['little_z']
        , 'norm_bound': kwargs['norm_bound']
        , 'corp_rate': kwargs['corp_rate']
        , 'ra': kwargs['ra']
        , 'mu': kwargs['mu']
        , 'vote_at': kwargs['vote_at']
        , 'sampled_only': kwargs['sampled_only']
        , 'vote_decay': kwargs['vote_decay']
        , 'vloss_eps': kwargs['vloss_eps']
        , 'eps_cof': kwargs['eps_cof']
        # , 'mlt': kwargs['mlt']
        , 'queue_size': kwargs['queue_size']
        , 'gm_iter': kwargs['gm_iter']
        , 'list_size': kwargs['list_size']
    }

    tModelList = {
        ('mclr', 'mnist'): Mclr_Logistic
        , ('dnn', 'mnist'): DNN
        , ('mclr', 'femnist'): Mclr_Logistic_Femnist
        , ('lr', 'femnist'): Mclr_CrossEntropy_Femnist # Logistic Regression model
        , ('dnn', 'femnist'): DNN_Femnist
        , ('cnn', 'femnist'): CNN_Femnist
        , ('mclr', 'fashion_mnist'): Mclr_Logistic
        , ('dnn', 'fashion_mnist'): DNN
        , ('cnn', 'cifar10'): CNNCifar2
        , ('lstm', 'sent140'): Sent140_LSTM
        , ('lstm', 'shakespeare'): Shkspr_LSTM
    }

    model = tModelList[(model_name, dataset)]().to(device)

    # algorithm Class
    tClassList = {
         'ListDec': ListDec
        , 'SigMdl': SigMdl
    }
    
    print(time.strftime("%Y-%m-%d %X", time.localtime()))
    start_t = time.time()
    print("=" * 80)
    for k in kwargs.keys():
        print(k + ':\t{}'.format(kwargs[k]))
    print("=" * 80)

    trainer = tClassList[name](device=device, name=name, model=model, dataset=dataset, hyperparams=hyperparams)
    trainer.train()
    print("Running Time: {:.04f} minutes".format((time.time() - start_t)/60.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion_mnist",
                        choices=["mnist", "femnist", "synthetic", "cifar10", "fashion_mnist", "sent140", "shakespeare"])
    parser.add_argument("--model_name", type=str, default="mclr"
                        , choices=["dnn", "mclr", "cnn", "lstm", "lr","cnn1"])
    parser.add_argument("--loss_name", type=str, default="NLLLoss"
                        , choices=["NLLLoss", "CrossEntropyLoss", "BCEWithLogitsLoss", "BCELoss", "mcFocalLoss", "balancedFocalLoss"])
    parser.add_argument("--optimizer_name", type=str, default="SGD"
                        , choices=["SGD", "Adam", "Adagrad", "pFedMeOptimizer", "CustomSGD"])
    parser.add_argument("--name", type=str, default="FedAvg"
                        , choices=["FedAvg", "ListDec", "SigMdl"])
    parser.add_argument("--batch_size", type=int, default=20
                        , help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-2
                        , help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0
                        , help="Average moving")
    # parser.add_argument("--lamda", type=float, default=15
    #                     , help="Regularization term")
    parser.add_argument("--total_epochs", type=int, default=1
                        , help="Total global iteration")
    parser.add_argument("--local_epochs", type=int, default=20
                        , help="Local iteration between aggregation")
    parser.add_argument("--num_aggregate_locals", type=int, default=20
                        , help="Number of Users per round")
    # parser.add_argument("--prox_iters", type=int, default=5
    #                     , help="Computation steps")
    # parser.add_argument("--personal_learning_rate", type=float, default=1e-2
    #                     , help="Personalized learning rate")
    parser.add_argument("--times", type=int, default=1
                        , help="Running time")
    parser.add_argument("--gpu", type=int, default=0
                        , help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    # parser.add_argument("--eta", type=float, default=5e-2
    #                     , help="Extra hyperparam")
    # parser.add_argument("--tau", type=float, default=1e-2
    #                     , help="Extra hyperparam")
    parser.add_argument("--at", type=str, default="clean"
                        , help="Byzantine attack", choices = ['clean', 'gauss', 'empire', 'labelflip', 'signflip', 'little', 'omniscient'])
    parser.add_argument("--empire_para", type=float, default=1.1
                        , help="hyperparam of empire attack")
    parser.add_argument("--little_z", type=float, default=20
                        , help='hyperparam of little attack')
    parser.add_argument("--corp_rate", type=float, default=0.6
                        , help="fraction of Byzantine locals")
    parser.add_argument("--ra", type=str, default="FedAvg"
                        , help="anti-Byzantine robust aggregator")
    parser.add_argument("--mu", type=float, default=0.0
                        , help="the momentum of global model")
    parser.add_argument("--local_data_distrb", type=str, default='niid'
                        , help="local data distribution", choices=['iid','niid'])
    parser.add_argument("--logf", type=str, default='results'
                        , help="log folder")
    parser.add_argument("--vote_at", type=str, default="worst"
                        , help="voting strategy of Byzantine nodes", choices=['worst','random'])
    parser.add_argument("--patience_iter", type=int, default=3000
                        , help="max round with on decrease in train loss")
    parser.add_argument("--eval_every", type=int, default=50
                        , help="evaluate global model every 10 rounds")
    parser.add_argument("--sampled_only", type=str, default='true' 
                        , help="only sampled locals train", choices=['true','false'])
    parser.add_argument("--vote_decay", type=float, default=1
                        , help="decay coefficient of old gmdl in voting phase")
    # parser.add_argument("--gamma", type=float, default=0.0
    #                     , help="Focal Loss")
    parser.add_argument('--valid_rate', type=float, default=0.2
                        , help='rate of validation set splited from training set')
    # parser.add_argument('--fl_beta', type=float, default=0.0
    #                     , help='beta in imbanlanced Cross Entropy Loss')
    parser.add_argument('--vloss_eps', type=float, default=1.0
                        , help = 'loss eps of new model in voting phase') 
    parser.add_argument('--eps_cof', type=float, default=0.0
                        , help = 'increase of eps every 2000 global round')
    # parser.add_argument('--mlt', type=str, default='false'
    #                     , help = 'multiple local training', choices=['true','false'])  
    parser.add_argument('--queue_size', type=int, default=0
                        , help = 'size of local queue of loss bound and cos similarity')
    parser.add_argument("--gm_iter", type=int, default=1
                        , help='number of iterations of GM robust aggregator') 
    parser.add_argument("--list_size", type=int, default=0
                        , help='the size of global model list')
    parser.add_argument("--norm_bound",type=str, default = 'fix'
                        , help = 'the bound of norm mean aggregator', choices=['adap', 'fix'])
    args = parser.parse_args()

    kwargs = {
        'gpu': args.gpu
        , 'name': args.name
        , 'dataset': args.dataset
        , 'valid_rate': args.valid_rate
        , 'model_name': args.model_name
        , 'loss_name': args.loss_name
        , 'optimizer_name': args.optimizer_name
        , 'total_epochs': args.total_epochs
        , 'local_epochs': args.local_epochs
        , 'learning_rate': args.learning_rate
        , 'num_aggregate_locals': args.num_aggregate_locals
        , 'batch_size': args.batch_size
        , 'beta': args.beta
        , 'times': args.times
        , 'at': args.at
        , 'empire_para': args.empire_para
        , 'little_z': args.little_z
        , 'norm_bound': args.norm_bound
        , 'corp_rate': args.corp_rate
        , 'ra': args.ra
        , 'local_data_distrb': args.local_data_distrb
        , 'logf': args.logf
        , 'vote_at': args.vote_at
        , 'patience_iter': args.patience_iter
        , 'eval_every': args.eval_every
        # , 'vote': args.vote
        , 'mu': args.mu
        , 'sampled_only': args.sampled_only
        , 'vote_decay': args.vote_decay
        # , 'gamma': args.gamma
        # , 'fl_beta': args.fl_beta
        , 'vloss_eps': args.vloss_eps
        , 'eps_cof': args.eps_cof
        # , 'mlt': args.mlt
        , 'queue_size': args.queue_size
        , 'gm_iter': args.gm_iter
        ,'list_size': args.list_size
    }

    main(**kwargs)

