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


class AverageEventKernelCRF(KNRM):
    io_group = 'joint_graph'

    def __init__(self, para, ext_data=None):
        super(AverageEventKernelCRF, self).__init__(para, ext_data)
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
        mtx_e_score = h_packed_data['mtx_score']

        ts_evm = h_packed_data['ts_evm']
        mtx_evm_score = h_packed_data['mtx_evm_score']

        ts_feature_all = h_packed_data['ts_feature']

        if ts_feature_all.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature_all.size()),
                          self.node_feature_dim)
        assert ts_feature_all.size()[-1] == self.node_feature_dim

        if mtx_e.size()[:2] != ts_feature_all.size()[:2]:
            logging.error(
                'e mtx and feature tensor shape do not match: %s != %s',
                json.dumps(mtx_e.size()), json.dumps(ts_feature_all.size()))
        assert mtx_e.size()[:2] == ts_feature_all.size()[:2]

        node_score = F.tanh(self.node_lr(ts_feature_all))

        h_mid_data = {
            "mtx_e": mtx_e,
            "mtx_score": mtx_e_score
        }
        knrm_res = super(AverageEventKernelCRF, self).forward(h_mid_data)

        mixed_knrm = torch.cat((knrm_res.unsqueeze(-1), node_score), -1)
        output = self.linear_combine(mixed_knrm).squeeze(-1)
        return output

    def save_model(self, output_name):
        logging.info('saving knrm embedding and linear weights to [%s]',
                     output_name)
        super(AverageEventKernelCRF, self).save_model(output_name)
        np.save(open(output_name + '.node_lr.npy', 'w'),
                self.node_lr.weight.data.cpu().numpy())
        np.save(open(output_name + '.linear_combine.npy', 'w'),
                self.linear_combine.weight.data.cpu().numpy())
