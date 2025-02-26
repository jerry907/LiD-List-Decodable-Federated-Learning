from sklearn.datasets import fetch_openml
import numpy as np
import random
from tqdm import trange
import json
import os
from collections import Counter

random.seed(1)
np.random.seed(1)
NUM_USERS = 35

# Setup directory for train/test data
train_dir = './data/train_full/'
test_dir = './data/test_full/'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get cifar10 dataset
cifar10 = fetch_openml(data_id=40927, data_home='./data', cache=False)
cifar10.data = np.array(cifar10.data)
cifar10.target = np.array(cifar10.target) 

cifar10_data = []
for i in trange(10):
    _cifar10_data = []
    for l in range(cifar10.target.shape[0]):
        if int(cifar10.target[l]) == i:
            _cifar10_data.append(cifar10.data[l])
    cifar10_data.append(_cifar10_data)
print("Number of samples for each label:\n", [len(v) for v in cifar10_data]) # [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]
print("Fraction of samples for each label:\n", [len(v)/len(cifar10.data) for v in cifar10_data]) # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Split data
# num_samples_per_user = (cifar10.data.shape[0] // 6 * 5) // NUM_USERS # use 5/6 fraction of raw data
# print(f'num_samples_per_user: {num_samples_per_user}') # cifar: 1428
# indices = np.arange(cifar10.data.shape[0])
# np.random.shuffle(indices)
# train_num_samples = (num_samples_per_user // 6) * 5
# test_num_samples = num_samples_per_user // 6 # 238

num_samples_per_user = (cifar10.data.shape[0]) // NUM_USERS # full data
print(f'num_samples_per_user: {num_samples_per_user}') # cifar: 
indices = np.arange(cifar10.data.shape[0])
np.random.shuffle(indices)
train_num_samples = (num_samples_per_user // 6) * 5
test_num_samples = num_samples_per_user // 6 # 

# Assign data to each user
for i in range(NUM_USERS):
    uname = str(i)
    start_idx = i * num_samples_per_user
    end_train_idx = i * num_samples_per_user + train_num_samples
    end_test_idx = (i + 1) * num_samples_per_user
    
    train_idx = indices[start_idx:end_train_idx]
    test_idx = indices[end_train_idx:end_test_idx]
    print(f"{i} train_num_samples: {len(train_idx)}, test_num_samples: {len(test_idx)}") # train_num_samples: 1190, test_num_samples: 238

    train_data = {'users': [], 'num_samples': [],'user_data': {} }
    test_data = {'users': [], 'num_samples': [],'user_data': {} }
    
    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': cifar10.data[train_idx].tolist(),
                                      'y': [int(label) for label in cifar10.target[train_idx].tolist()]}
    train_data['num_samples'].append(len(train_data['user_data'][uname]['x']))
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': cifar10.data[test_idx].tolist(),
                                     'y': [int(label) for label in cifar10.target[test_idx].tolist()]}
    test_data['num_samples'].append(len(test_data['user_data'][uname]['x']))
    
    # user_data_counter = []
    # for i in trange(10):
    #     user_data_counter[i] = sum(train_data['user_data'][uname]['y'] == i)
    #     user_data_counter[i] += sum(test_data['user_data'][uname]['y'] == i)
    # print(f"user {uname} Number of samples for each label: {user_data_counter}")
    # print(f"user {uname} Fraction of samples for each label:\n", [c/sum(user_data_counter) for c in user_data_counter])
    print(f'user {uname} Counter for train data: ', Counter(train_data['user_data'][uname]['y']))
    print(f'user {uname} Counter for test data: ', Counter(test_data['user_data'][uname]['y']))
    
    # Save data for each user
    train_filename = os.path.join(train_dir, f'cifar10_train_user_{uname}.json')
    with open(train_filename, 'w') as outfile:
        json.dump(train_data, outfile)

    test_filename = os.path.join(test_dir, f'cifar10_test_user_{uname}.json')
    with open(test_filename, 'w') as outfile:
        json.dump(test_data, outfile)

print(f'end_test_idx {end_test_idx} less than total number of data {cifar10.data.shape[0]}')
print("Finish Generating IID Samples")
