import json
import os
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import one_hot
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3
NUM_CLASSES = {
    "femnist": 62,
    "cifar10": 10
}

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x) // batch_size + 1
    if (len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx * batch_size
        if (sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index + batch_size], data_y[sample_index: sample_index + batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_cifa_data():
    '''same as rfl/data/cifar10/generate_niid_100users.py, not called by framework'''
    print('framework/data_utils.py called read_cifa_data()!')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader, 0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 20  # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label == i
        cifa_data.append(cifa_data_image[idx])

    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l] + 10].tolist()
            y[user] += (l * np.ones(10)).tolist()
            idx[l] += 10

    print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v) - NUM_USERS]] for v in cifa_data]) * \
            props / np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    # props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    # idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user // int(NUM_USERS / 10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples) + numran1  # + 200
            if (NUM_USERS <= 20):
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l] + num_samples].tolist()
                y[user] += (l * np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len os user:", user, j,
                      "len data", len(X[user]), num_samples)

    print("IDX2:", idx)  # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len

        # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\

        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X[i][:test_len], 'y': y[i][:test_len]}
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] = {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

    return train_data['users'], _, train_data['user_data'], test_data['user_data']


def read_data(dataset, root='', data_distrb=''):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    train_data_dir = os.path.join(root + 'data', dataset, 'data', 'train') # data path of iid data_distrb
    test_data_dir = os.path.join(root + 'data', dataset, 'data', 'test')
    if data_distrb == 'niid':
        train_data_dir = os.path.join(root + 'data', dataset, 'data', 'train_niid')
        test_data_dir = os.path.join(root + 'data', dataset, 'data', 'test_niid')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata: # key in dic: check if the key in dic.keys() by default
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))
    # print('data_utils, read_data,clients: ', clients)
    # clients:  ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', \
    # '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '4', '5', '6', '7', '8', '9']

    if(dataset == "femnist") or dataset == "cifar10":
        train_data, test_data = global_noml(train_data, test_data)

    return clients, groups, train_data, test_data

def read_user_data(index, data, dataset, hparams):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = np.array(train_data['x']), np.array(train_data['y']), np.array(test_data['x']), np.array(test_data['y'])
    if (dataset == "mnist" or dataset == "fmnist" or dataset == "fashion_mnist" or dataset == "femnist"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif (dataset == "cifar10"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif (dataset == "sent140"):
        X_train = torch.Tensor(X_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.int64)
        if hparams['loss_name'] == 'BCELoss' or hparams['loss_name'] == 'BCEWithLogitsLoss':
            y_train = one_hot(torch.Tensor(y_train).type(torch.int64), num_classes=2).type(torch.float32)
            y_test = one_hot(torch.Tensor(y_test).type(torch.int64), num_classes=2).type(torch.float32)
        elif hparams['loss_name'] == 'CrossEntropyLoss' or hparams['loss_name'] == 'NLLLoss':
            y_train = torch.Tensor(y_train).type(torch.int64)
            y_test = torch.Tensor(y_test).type(torch.int64)
    elif (dataset == "shakespeare"):
        X_train = torch.Tensor(X_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.int64)
        if hparams['loss_name'] == 'BCELoss' or hparams['loss_name'] == 'BCEWithLogitsLoss':
            y_train = one_hot(torch.Tensor(y_train).type(torch.int64), num_classes=45).type(torch.float32)
            y_test = one_hot(torch.Tensor(y_test).type(torch.int64), num_classes=45).type(torch.float32)
        elif hparams['loss_name'] == 'CrossEntropyLoss' or hparams['loss_name'] == 'NLLLoss':
            y_train = torch.Tensor(y_train).type(torch.int64)
            y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

def read_user_data_vad(index, data, dataset, hparams):
    valid_rate=hparams['valid_rate']
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_test, y_test = np.array(test_data['x']), np.array(test_data['y'])

    # split train data into validation set and training set in the ratio 80:20
    data = list(zip(train_data['x'], train_data['y']))
    rng = random.Random(id)
    rng.shuffle(data)
    split_point = int((1-valid_rate) * len(data))
    X_train, y_train = zip(*data[:split_point])
    X_valid, y_valid = [], []
    if valid_rate > 0:
        X_valid, y_valid = zip(*data[split_point:])
    X_train, y_train, X_valid, y_valid = np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid)

    if (dataset == "mnist" or dataset == "fmnist" or dataset == "fashion_mnist" or dataset == "femnist"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        X_valid = torch.Tensor(X_valid).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_valid = torch.Tensor(y_valid).type(torch.int64)
    elif (dataset == "cifar10"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        X_valid = torch.Tensor(X_valid).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_valid = torch.Tensor(y_valid).type(torch.int64)
    elif (dataset == "sent140"):
        X_train = torch.Tensor(X_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.int64)
        X_valid = torch.Tensor(X_valid).type(torch.int64)
        if hparams['loss_name'] == 'BCELoss' or hparams['loss_name'] == 'BCEWithLogitsLoss':
            y_train = one_hot(torch.Tensor(y_train).type(torch.int64), num_classes=2).type(torch.float32)
            y_test = one_hot(torch.Tensor(y_test).type(torch.int64), num_classes=2).type(torch.float32)
            y_valid = one_hot(torch.Tensor(y_valid).type(torch.int64), num_classes=2).type(torch.float32)
        elif hparams['loss_name'] == 'CrossEntropyLoss' or hparams['loss_name'] == 'NLLLoss':
            y_train = torch.Tensor(y_train).type(torch.int64)
            y_test = torch.Tensor(y_test).type(torch.int64)
            y_valid = one_hot(torch.Tensor(y_valid).type(torch.int64), num_classes=2).type(torch.float32)
    elif (dataset == "shakespeare"):
        X_train = torch.Tensor(X_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.int64)
        X_valid = torch.Tensor(X_valid).type(torch.int64)
        if hparams['loss_name'] == 'BCELoss' or hparams['loss_name'] == 'BCEWithLogitsLoss':
            y_train = one_hot(torch.Tensor(y_train).type(torch.int64), num_classes=45).type(torch.float32)
            y_test = one_hot(torch.Tensor(y_test).type(torch.int64), num_classes=45).type(torch.float32)
            y_valid = one_hot(torch.Tensor(y_valid).type(torch.int64), num_classes=45).type(torch.float32)
        elif hparams['loss_name'] == 'CrossEntropyLoss' or hparams['loss_name'] == 'NLLLoss':
            y_train = torch.Tensor(y_train).type(torch.int64)
            y_test = torch.Tensor(y_test).type(torch.int64)
            y_valid = torch.Tensor(y_valid).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        X_test = torch.Tensor(X_test).type(torch.float32)
        X_valid = torch.Tensor(X_valid).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        y_test = torch.Tensor(y_test).type(torch.int64)
        y_valid = torch.Tensor(y_valid).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    valid_data = [(x, y) for x, y in zip(X_valid, y_valid)]
    return id, train_data, test_data, valid_data

def global_noml(train_data, test_data):
    '''Z-Score Normalization, x = x-miu/sigma'''
    print('Global Data Z-Score Normalization')
    tdata_lst = []
    for user,tdata in train_data.items():
        tdata_lst.extend(tdata["x"])
    avg_tdata = np.mean(tdata_lst, axis = 0)
    std_tdata = np.std(tdata_lst, axis = 0)
    std_tdata += np.array([1e-6]*len(std_tdata)) # avoid devided by 0
    # print('len(avg_tdata): {0}, len(std_tdata): {1}'.format(len(avg_tdata), len(std_tdata))) # len(avg_tdata): 784, len(std_tdata): 784
    
    for user,tdata in train_data.items():
        # for i in range(len(tdata["x"])):[i]
        train_data[user]["x"] = (train_data[user]["x"] - avg_tdata) / std_tdata
    for user,test_d in test_data.items():
        # for i in range(len(test_d["x"])):[i]
        test_data[user]["x"] = (test_data[user]["x"] - avg_tdata) / std_tdata
    
    return train_data, test_data

def tData_lf(train_data,dataset,bs=32):
    '''apply labelflip to train data verified OK'''
    m = NUM_CLASSES[dataset]
    newx, newy = [], []
    for x, y in train_data: 
        newx.extend(x)
        newy.extend(torch.Tensor(m - 1. - y).type(y.dtype))

    return DataLoader([(xi, yi) for xi, yi in zip(newx, newy)], bs)