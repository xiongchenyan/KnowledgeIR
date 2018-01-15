"""
crf a-like models
node and edge

KernelCRF:
    node is input node feature
    edge is the embedding KNRM
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling
from knowledge4ir.salience.knrm_vote import KNRM
import logging
import json
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()


class KernelCRF(KNRM):
    io_group = 'feature'

    def __init__(self, para, ext_data=None):
        super(KernelCRF, self).__init__(para, ext_data)
        self.node_feature_dim = para.node_feature_dim
        self.node_lr = nn.Linear(self.node_feature_dim, 1, bias=False)
        if use_cuda:
            self.node_lr.cuda()
        return

    def forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        ts_feature = h_packed_data['ts_feature']

        if ts_feature.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature.size()), self.node_feature_dim)
        assert ts_feature.size()[-1] == self.node_feature_dim
        if mtx_e.size()[:2] != ts_feature.size()[:2]:
            logging.error(
                'e mtx and feature tensor shape do not match: %s != %s',
                json.dumps(mtx_e.size()), json.dumps(ts_feature.size()))
        assert mtx_e.size()[:2] == ts_feature.size()[:2]

        node_score = F.relu(self.node_lr(ts_feature)).squeeze(-1)
        h_mid_data = {
            "mtx_e": mtx_e,
            "mtx_score": node_score
        }
        output = super(KernelCRF, self).forward(h_mid_data)
        return output

    def save_model(self, output_name):
        logging.info('saving knrm embedding and linear weights to [%s]',
                     output_name)
        super(KernelCRF, self).save_model(output_name)
        np.save(open(output_name + '.node_lr.npy', 'w'),
                self.node_lr.weight.data.cpu().numpy())


class LinearKernelCRF(KNRM):
    io_group = 'feature'

    def __init__(self, para, ext_data=None):
        super(LinearKernelCRF, self).__init__(para, ext_data)
        self.node_feature_dim = para.node_feature_dim
        self.node_lr = nn.Linear(self.node_feature_dim, 1, bias=False)
        logging.info('node feature dim %d', self.node_feature_dim)
        self.linear_combine = nn.Linear(2, 1)
        if use_cuda:
            self.node_lr.cuda()
            self.linear_combine.cuda()
        return

    def forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        ts_feature = h_packed_data['ts_feature']

        if ts_feature.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature.size()), self.node_feature_dim)
        assert ts_feature.size()[-1] == self.node_feature_dim
        if mtx_e.size()[:2] != ts_feature.size()[:2]:
            logging.error(
                'e mtx and feature tensor shape do not match: %s != %s',
                json.dumps(mtx_e.size()), json.dumps(ts_feature.size()))
        assert mtx_e.size()[:2] == ts_feature.size()[:2]

        print ts_feature
        print self.node_lr(ts_feature)
        import sys
        sys.stdin.readline()

        node_score = F.tanh(self.node_lr(ts_feature))
        mtx_score = ts_feature.narrow(-1, 0, 1).squeeze(
            -1)  # frequency is the first dim of feature, always
        h_mid_data = {
            "mtx_e": mtx_e,
            "mtx_score": mtx_score
        }
        knrm_res = super(LinearKernelCRF, self).forward(h_mid_data)

        mixed_knrm = torch.cat((knrm_res.unsqueeze(-1), node_score), -1)
        output = self.linear_combine(mixed_knrm).squeeze(-1)
        return output

    def save_model(self, output_name):
        logging.info('saving knrm embedding and linear weights to [%s]',
                     output_name)
        super(LinearKernelCRF, self).save_model(output_name)
        np.save(open(output_name + '.node_lr.npy', 'w'),
                self.node_lr.weight.data.cpu().numpy())
        np.save(open(output_name + '.linear_combine.npy', 'w'),
                self.linear_combine.weight.data.cpu().numpy())
