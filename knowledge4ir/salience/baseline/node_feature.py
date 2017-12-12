"""
lr on entity embedding
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from knowledge4ir.salience.base import SalienceBaseModel

use_cuda = torch.cuda.is_available()


class FeatureLR(SalienceBaseModel):
    io_group = 'feature'
    def __init__(self, para, ext_data=None):
        super(FeatureLR, self).__init__(para, ext_data)
        self.node_feature_dim = para.node_feature_dim
        self.l_hidden_dim = para.l_hidden_dim
        last_dim = self.node_feature_dim
        self.l_node_lr = nn.ModuleList()
        for hidden_d in self.l_hidden_dim:
            this_linear = nn.Linear(last_dim, hidden_d, bias=False)
            self.l_node_lr.append(this_linear)
            last_dim = hidden_d
        self.l_node_lr.append(nn.Linear(last_dim, 1, bias=False))

        if use_cuda:
            for linear in self.l_node_lr:
                linear.cuda()

    def forward(self, h_packed_data):
        assert 'ts_feature' in h_packed_data
        ts_feature = h_packed_data['ts_feature']
        middle = ts_feature
        for linear in self.l_node_lr:
            middle = F.tanh(linear(middle))
        output = middle.squeeze(-1)
        return output

    def save_model(self, output_name):
        logging.info('saving node feature linear weights to [%s]', output_name)
        for p in xrange(len(self.l_node_lr)):
            np.save(open(output_name + '.linear_%d.npy' % p, 'w'),
                    self.l_node_lr[p].weight.data.cpu().numpy())


class FrequencySalience(SalienceBaseModel):

    def __init__(self, para, ext_data=None):
        super(FrequencySalience, self).__init__(para, ext_data)
        # self.linear = nn.Linear(1, 1)
        # if use_cuda:
        #     self.linear.cuda()
        return

    def forward(self, h_packed_data,):
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        output = mtx_score
        # output = mtx_score.unsqueeze(-1)
        # output = self.linear(output)
        # output = output.squeeze(-1)
        if use_cuda:
            return output.cuda()
        else:
            return output