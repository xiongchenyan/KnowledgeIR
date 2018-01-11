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


class StructEventKernelCRF(KNRM):
    def forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        ts_evm = h_packed_data['ts_evm']
        v_evm_length = h_packed_data['v_evm_length']

        ts_feature_all = h_packed_data['ts_feature']

        mtx_score = h_packed_data['mtx_score']

        print ts_evm.size()

        print v_evm_length

        print mtx_e.size()[:2]

        print ts_feature_all.size()[:2]

        import sys
        sys.stdin.readline()

        mtx_evm = self.event_embedding(ts_evm, v_evm_length)

        combined_mtx_e = mtx_e + mtx_evm

        if ts_feature_all.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature_all.size()),
                          self.node_feature_dim)
        assert ts_feature_all.size()[-1] == self.node_feature_dim

        if combined_mtx_e.size()[:2] != ts_feature_all.size()[:2]:
            logging.error(
                'e mtx and feature tensor shape do not match: %s != %s',
                json.dumps(combined_mtx_e.size()),
                json.dumps(ts_feature_all.size()))
        assert combined_mtx_e.size()[:2] == ts_feature_all.size()[:2]

        node_score = F.tanh(self.node_lr(ts_feature_all))

        knrm_res = self.forward_kernel_with_embedding(combined_mtx_e, mtx_score)

        mixed_knrm = torch.cat((knrm_res.unsqueeze(-1), node_score), -1)
        output = self.linear_combine(mixed_knrm).squeeze(-1)
        return output

    def event_embedding(self, ts_evm, v_evm_length):
        raise NotImplementedError

    def forward_kernel_with_embedding(self, mtx_embedding, mtx_score):
        kp_mtx = self._kernel_scores(mtx_embedding, mtx_score)
        output = self.linear(kp_mtx)
        output = output.squeeze(-1)
        return output

    def save_model(self, output_name):
        logging.info('saving knrm embedding and linear weights to [%s]',
                     output_name)
        super(StructEventKernelCRF, self).save_model(output_name)
        np.save(open(output_name + '.node_lr.npy', 'w'),
                self.node_lr.weight.data.cpu().numpy())
        np.save(open(output_name + '.linear_combine.npy', 'w'),
                self.linear_combine.weight.data.cpu().numpy())


class AverageEventKernelCRF(StructEventKernelCRF):
    io_group = 'joint_graph'

    def __init__(self, para, ext_data=None):
        super(StructEventKernelCRF, self).__init__(para, ext_data)

    def event_embedding(self, ts_evm, v_evm_length):
        pass
