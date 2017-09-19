"""
simple baseline model
:frequency
"""


import logging

import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()


class FrequencySalience(nn.Module):

    def __init__(self, layer, vocab_size, embedding_dim, pre_embedding=None):
        super(FrequencySalience, self).__init__()
        self.linear = nn.Linear(1, 1)
        if use_cuda:
            self.linear.cuda()
        return

    def forward(self, mtx_e, mtx_score):
        output = mtx_score.unsqueeze(-1)
        output = self.linear(output)
        output = output.squeeze(-1)
        if use_cuda:
            return output.cuda()
        else:
            return output
