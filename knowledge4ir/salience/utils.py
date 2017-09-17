"""
utils for salience inference
"""

import torch
from torch import nn
from torch.autograd import Variable


def hinge_loss(output, target):
    assert not target.requires_grad
    print output.size()
    print target.size()
    loss = torch.mm(target.type(torch.cuda.FloatTensor), (target.type(torch.cuda.FloatTensor) - output))
    loss = loss.clamp(min=0).sum()
    return loss

