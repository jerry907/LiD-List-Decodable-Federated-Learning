# LiD-FL: Towards List-Decodable Federated Learning

This repository provides source code to reproduce the experimental results in the paper "LiD: List-Decodable Federated Learning". This project is based on a fork of the [Leaf](leaf.cmu.edu) benchmark suite and [pFedBreD_public](https://github.com/BDeMo/pFedBreD_public).

If you use this code, please cite the paper using the bibtex reference below

```
@article{DBLP:journals/corr/abs-2408-04963,
  author       = {Hong Liu and
                  Liren Shan and
                  Han Bao and
                  Ronghui You and
                  Yuhao Yi and
                  Jiancheng Lv},
  title        = {LiD-FL: Towards List-Decodable Federated Learning},
  journal      = {CoRR},
  volume       = {abs/2408.04963},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2408.04963},
  doi          = {10.48550/ARXIV.2408.04963},
  eprinttype    = {arXiv},
  eprint       = {2408.04963},
  timestamp    = {Tue, 17 Sep 2024 08:35:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2408-04963.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Abstract
-----------------
Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest clients, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Assuming the variance of gradient noise in stochastic gradient descent is bounded, we prove a convergence theorem of our method for strongly convex and smooth losses. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.

The [accompanying paper](https://arxiv.org/abs/2408.04963).


Installation                                                                                                                   
-----------------
This code is written in Python 3.8 and has been tested on PyTorch 1.4+.
A conda environment file is provided in `lidfl.yml` with all dependencies except PyTorch. 
It can be installed by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
as follows

```
conda env create -f lidfl.yml 
```

**Installing PyTorch:** Instructions to install a PyTorch compatible with the CUDA on your GPUs (or without GPUs) can be found [here](https://pytorch.org/get-started/locally/).


Dataset
-----------
Due to the large size of dataset, we ddi not upload dataset here, and the complete dataset is available at 
https://drive.google.com/file/d/1_FopTqq_ZgVO7OkNBtCiw9SyTyGvU5m6/view?usp=drive_link.
1. ### FEMNIST 

  * **Overview:** Character Recognition Dataset
  * **Original dataset:** 62 classes (10 digits, 26 lowercase, 26 uppercase), 3500 total users.(leaf.cmu.edu)
  * **Preprocess:** We sample 5% of the images in the original dataset to construct our datasets. For the homogeneous setting, each client sample images from a uniform distribution over 62 classes.   We generate heterogeneous datasets for clients using categorical distributions qmdrawn from a Dirichlet distribution q ∼ Dir(αp), where p is a prior class distribution over 62 classes (Hsu, Qi, and Brown 2019). Each client sample from a categorical distribution characterized by an independent q . In our experiment for the heterogeneous setting, we let α = 0.1, which is described as the extreme heterogeneity setting in (Allouah et al. 2023a).
  * **Task:** Image Classification
  * **Directory:** ```data/femnist``` 

2. ### CIFAR10

  * **Overview:** Tiny Images Dataset
  * **Original dataset:** 60000 32x32 colour images in 10 classes, with 6000 images per class.(https://www.cs.toronto.edu/~kriz/cifar.html)
  * **Preprocess:** We use a small dataset of 35 clients uniformly sampled from the CIFAR-10 dataset, and each client contains 300 train samples and 60 test samples.
  * **Task:** Image Classification
  * **Directory:** ```data/cifar10``` 


Reproducing Experiments in the Paper
-------------------------------------

As the data has been set up, the scripts provided in the folder ```scripts/``` can be used 
to reproduce the experiments in the paper.

Change directory to ```main_fl.py``` and run the scripts as 
```
./scripts/femnist_cnn/run.sh  
```
