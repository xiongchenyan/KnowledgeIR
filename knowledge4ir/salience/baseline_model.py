"""
simple baseline model
:frequency
"""


import logging

import torch
import torch.nn as nn
from knowledge4ir.salience.utils import SalienceBaseModel
use_cuda = torch.cuda.is_available()


class FrequencySalience(SalienceBaseModel):

    def __init__(self, para, pre_embedding=None):
        super(FrequencySalience, self).__init__(para, pre_embedding)
        self.linear = nn.Linear(1, 1)
        if use_cuda:
            self.linear.cuda()
        return

    def forward(self, h_packed_data,):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        output = mtx_score.unsqueeze(-1)
        output = self.linear(output)
        output = output.squeeze(-1)
        if use_cuda:
            return output.cuda()
        else:
            return output
