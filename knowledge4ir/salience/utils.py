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


def p_at_k(l_score, l_label, k):
    l_d = zip(l_score, l_label)
    l_d.sort(key=lambda item: -item[0])
    correct = 0
    for score, label in l_d[:k]:
        if label > 0:
            correct += 1
    return float(correct) / k
