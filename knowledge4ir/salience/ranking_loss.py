"""
ranking loss
classification hinge
pairwise loss
"""
import torch
use_cuda = torch.cuda.is_available()


def _assert(output, target):
    assert not target.requires_grad
    assert output.size() == target.size()


def hinge_loss(output, target):
    _assert(output, target)
    if use_cuda:
        loss = target.type(torch.cuda.FloatTensor) * (target.type(torch.cuda.FloatTensor) - output)
    else:
        loss = target.type(torch.FloatTensor) * (target.type(torch.FloatTensor) - output)
    loss = loss.clamp(min=0).mean()
    return loss


def pairwise_loss(output, target):
    """
    output's last dimension is ranking score
    target's last dimension is ranking label
    max(0, 1 - d^+ + d^-)
    -> max(0, (target_i - target_j) * [(target_i - target_j) - (output_i - output_j)]
    the key is to make pairwise difference in via output or target's last dimension,
    then it is simple hinge_loss
    :param output:
    :param target:
    :return:
    """
    _assert(output, target)
    pairwise_output = _pair_diff(output)
    pairwise_target = _pair_diff(target)
    pairwise_target = pairwise_target.sign()
    return hinge_loss(pairwise_output, pairwise_target)


def _pair_diff(ts_score):
    """
    score's i - j in the last dimension
    :param ts_score:
    :return:
    """
    mid_score = ts_score.unsqueeze(-1)
    mid_score = mid_score.expand(
        mid_score.size()[:-1] + (mid_score.size()[-2],)
    )
    return mid_score.transpose(-2, -1)
