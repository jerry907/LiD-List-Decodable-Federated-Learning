from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss, BCELoss
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from balanced_loss import Loss

def get_loss(loss_name):
    if loss_name == 'CrossEntropyLoss':
        return CrossEntropyLoss
    elif loss_name == 'NLLLoss':
        return NLLLoss
    elif loss_name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss
    elif loss_name == 'BCELoss': # binary classification
        return BCELoss
    elif loss_name == 'mcFocalLoss':
        return FocalLoss
    elif loss_name == 'balancedFocalLoss':
        return Loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1) # N => N,1, the size -1 is inferred from other dimensions

        logpt = F.log_softmax(input, dim=1) 
        # after log_softmax, logpt.shape:  torch.Size([32, 62]), target.shape  torch.Size([32, 1])
        logpt = logpt.gather(1,target)
        # after gather, logpt.shape:  torch.Size([32, 1])
        logpt = logpt.view(-1) # N,1 => N
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()