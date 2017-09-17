"""
utils for salience inference
"""

import torch
from torch import nn
from torch.autograd import Variable


def hinge_loss(output, target):
    assert not target.requires_grad
    assert output.size() == target.size()
    loss = target.type(torch.cuda.FloatTensor) * (target.type(torch.cuda.FloatTensor) - output)
    loss = loss.clamp(min=0).mean()
    return loss

