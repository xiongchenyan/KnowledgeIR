"""
utils for salience inference
"""

import torch
from torch import nn
from torch.autograd import Variable


def hinge_loss(output, target):
    assert not target.requires_grad
    loss = torch.mm(target.type(torch.FloatTensor), (target.type(torch.FloatTensor) - output))
    loss = loss.clamp(min=0).sum()
    return loss

