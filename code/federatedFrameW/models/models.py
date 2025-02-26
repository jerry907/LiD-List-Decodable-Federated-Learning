import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from federatedFrameW.utils.language_utils import read_embedding


class Shkspr_LSTM(nn.Module):
    def __init__(self, hidden_size=50, d_embed=50, s_vocab=18444):
        super(Shkspr_LSTM, self).__init__()
        # self.emb = nn.Embedding(s_vocab, d_embed, padding_idx=1)
        # self.emb.requires_grad_(False)

        embedings = torch.FloatTensor(np.array(read_embedding('shakespeare')))
        self.emb = nn.Embedding.from_pretrained(embedings, freeze=True, padding_idx=1)

        self.encoder = nn.LSTM(input_size=d_embed, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 46)
        # self.fc2 = nn.Linear(hidden_size * 2, 46)

    def forward(self, x):
        self.encoder.flatten_parameters()
        embeding = self.emb(x)
        # print(embeding[0, :, :])
        out, (hidden, _) = self.encoder(embeding)
        # encoding = torch.cat((torch.sum(out, dim=1) / out.shape[1], torch.sum(hidden, dim=0) / out.shape[0]), -1)
        encoding = torch.cat((out[:, -1, :], hidden[-1, :, :]), -1)
        # encoding = out[:, -1, :]
        # return F.log_softmax(self.fc2(encoding), dim=1)
        output = F.log_softmax(self.fc2(F.leaky_relu(self.fc1(encoding), negative_slope=0.1)), dim=1)
        return output


class Sent140_LSTM(nn.Module):
    def __init__(self, hidden_size=50, d_embed=50, s_vocab=22764):
        super(Sent140_LSTM, self).__init__()
        # self.emb = nn.Embedding(s_vocab, d_embed, padding_idx=1)
        # self.emb.requires_grad_(False)

        embedings = torch.FloatTensor(read_embedding('sent140'))
        self.emb = nn.Embedding.from_pretrained(embedings, freeze=True, padding_idx=0)

        self.encoder = nn.LSTM(input_size=d_embed, hidden_size=hidden_size, num_layers=2, batch_first=True,
                               bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size * 100)
        # self.fc1 = nn.Linear(hidden_size, hidden_size * 100)
        self.fc2 = nn.Linear(hidden_size * 100, 2)
        # self.fc2 = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        self.encoder.flatten_parameters()
        embeding = self.emb(x)
        out, (hidden, _) = self.encoder(embeding)
        # encoding = torch.cat((torch.sum(out, dim=1) / out.shape[1], torch.sum(hidden, dim=0) / out.shape[0]), -1)
        encoding = torch.cat([out[:, -1, :50], out[:, -1, 50:], hidden[-1, :, :], hidden[-2, :, :]], -1)
        # encoding = hidden[-1, :, :]
        # return F.log_softmax(self.fc2(encoding), dim=1)
        output = F.log_softmax(self.fc2(F.leaky_relu(self.fc1(encoding), negative_slope=0.1)), dim=-1)
        # print('output', F.log_softmax(output[0]))
        return output


class Sent140_RNN(nn.Module):
    def __init__(self, hidden_size=128, d_embed=100, s_vocab=87093):
        super(Sent140_RNN, self).__init__()
        self.emb = nn.Embedding(s_vocab, d_embed)
        self.encoder = nn.RNN(input_size=d_embed, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, h):
        x = torch.transpose(x, 0, 1)
        out, hidden = self.encoder(self.emb(x), h)
        return self.fc(hidden.squeeze(0))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        # torch.nn.init.xavier_uniform(self.fc1.weight.data)

    def forward(self, x):
        x = torch.flatten(x, 1).type(torch.float32)
        # print(x.shape, self.fc1)
        x = self.fc1(x)
        # t = [i for i in self.fc1.parameters()][0]
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_Logistic_Femnist(nn.Module):
    def __init__(self, input_dim=784, output_dim=62):
        super(Mclr_Logistic_Femnist, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        # weight initialization
        # w = torch.zeros(size=(output_dim, input_dim), requires_grad=True)
        # self.fc1.weight = Parameter(w) # type transform https://blog.csdn.net/D_handsome/article/details/122715621
        print('Mclr_Logistic, initial weight\n', self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc1.weight.data)

    def forward(self, x):
        x = torch.flatten(x, 1).type(torch.float32)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy_Femnist(nn.Module):
    def __init__(self, input_dim=784, output_dim=62):
        super(Mclr_CrossEntropy_Femnist, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # before flatten, x.shape: torch.Size([32, 1, 28, 28])
        x = torch.flatten(x, 1).type(torch.float32)
        outputs = self.linear(x)
        # after flatten, x.shape: torch.Size([32, 784]), outputs.shape: torch.Size([32, 62])
        return outputs


class DNN(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class DNN_Femnist(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=62):
        super(DNN_Femnist, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class CNN_Femnist(nn.Module):
    # model related to the one from 
    # https://github.com/krishnap25/RFA/blob/master/models/femnist/cnn.py
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 32, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(32, 64, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))

        self.fc = nn.Linear(1024, 62)

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return F.log_softmax(output, dim=1)

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        print('Create model CifarNet')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar2(nn.Module):
    '''Refer: Byzantine-Robust Federated Learning: Impact of Client Subsampling and Local Updates'''
    def __init__(self):
        super(CNNCifar2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=128),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            nn.LogSoftmax(dim=1)
        )
        print('Create Model CNNCifar2!')
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class CNNCifar1(nn.Module):
    # model related to the one from 
    # https://github.com/krishnap25/RFA/blob/master/models/femnist/cnn.py
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 32, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(32, 64, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))

        self.fc = nn.Linear(1600, 10)
        print('Create model CNN_Cifar')
        ## old:
        #self.fc = nn.Sequential(OrderedDict([
        #    ('f5', nn.Linear(1024, 256)),
        #    ('relu6', nn.ReLU()),
        #    ('f7', nn.Linear(256, 62)),
        #]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return F.log_softmax(output, dim=1)


class CifarNet2(nn.Module):
    def __init__(self):
        super(CifarNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return F.softmax(x, dim=1)


#################################
##### Neural Network trainmodel #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# class CNNCifar(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         self.weight_keys = [['fc1.weight', 'fc1.bias'],
#                             ['fc2.weight', 'fc2.bias'],
#                             ['fc3.weight', 'fc3.bias'],
#                             ['conv2.weight', 'conv2.bias'],
#                             ['conv1.weight', 'conv1.bias'],
#                             ]

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
