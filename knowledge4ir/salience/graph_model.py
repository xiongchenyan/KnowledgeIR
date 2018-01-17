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
    def __init__(self, para, ext_data=None):
        super(StructEventKernelCRF, self).__init__(para, ext_data)

        self.embedding_dim = para.embedding_dim
        self.node_feature_dim = para.node_feature_dim
        self.node_lr = nn.Linear(self.node_feature_dim, 1, bias=False)
        logging.info('node feature dim %d', self.node_feature_dim)
        self.linear_combine = nn.Linear(2, 1)

        if use_cuda:
            self.node_lr.cuda()
            self.linear_combine.cuda()

    def forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        mtx_evm = h_packed_data['mtx_evm']
        ts_args = h_packed_data['ts_args']
        mtx_arg_length = h_packed_data['mtx_arg_length']
        ts_arg_mask = h_packed_data['ts_arg_mask']

        ts_feature = h_packed_data['ts_feature']
        mtx_score = h_packed_data['mtx_score']

        mtx_e_embedding = self.embedding(mtx_e)

        if mtx_evm is None:
            # For documents without events.
            combined_mtx_e = mtx_e_embedding
        else:
            mtx_evm_embedding = self.event_embedding(mtx_evm, ts_args,
                                                     mtx_arg_length,
                                                     ts_arg_mask)
            combined_mtx_e = torch.cat((mtx_e_embedding, mtx_evm_embedding), 1)

        if ts_feature.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature.size()), self.node_feature_dim)
        assert ts_feature.size()[-1] == self.node_feature_dim

        if combined_mtx_e.size()[:2] != ts_feature.size()[:2]:
            logging.error(
                'First 2 dimension of e mtx and feature tensor do not '
                'match: %s != %s',
                json.dumps(combined_mtx_e.size()),
                json.dumps(ts_feature.size()))
        assert combined_mtx_e.size()[:2] == ts_feature.size()[:2]

        node_score = F.tanh(self.node_lr(ts_feature))

        knrm_res = self.forward_kernel_with_embedding(combined_mtx_e, mtx_score)

        mixed_knrm = torch.cat((knrm_res.unsqueeze(-1), node_score), -1)
        output = self.linear_combine(mixed_knrm).squeeze(-1)
        return output

    def event_embedding(self, mtx_evm, ts_args, mtx_arg_length, ts_arg_mask):
        raise NotImplementedError

    def _argument_sum(self, ts_args, ts_arg_mask):
        l_evm_embedding = []

        for mtx_args, mask in zip(ts_args, ts_arg_mask):
            mtx_args_embedding = self.embedding(mtx_args)
            masked_embedding = mtx_args_embedding * mask.unsqueeze(2)
            arg_embedding_sum = masked_embedding.sum(1)
            l_evm_embedding.append(arg_embedding_sum)

        return torch.stack(l_evm_embedding)

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
        super(AverageEventKernelCRF, self).__init__(para, ext_data)

    def event_embedding(self, mtx_evm, ts_args, mtx_arg_length, ts_arg_mask):
        mtx_p_embedding = self.embedding(mtx_evm)

        if ts_args is None:
            # When there are no arguments, the embedding is just the predicate.
            mtx_evm_embedding_aver = mtx_p_embedding
        else:
            mtx_arg_embedding_sum = self._argument_sum(ts_args, ts_arg_mask)
            mtx_evm_embedding_sum = mtx_p_embedding + mtx_arg_embedding_sum

            # aver = (embedding sum) / (1 + arg length)
            mtx_full_length = (mtx_arg_length + 1).type_as(
                mtx_evm_embedding_sum).unsqueeze(2)
            mtx_evm_embedding_aver = mtx_evm_embedding_sum / mtx_full_length
        return mtx_evm_embedding_aver


class AverageArgumentKernelCRF(StructEventKernelCRF):
    io_group = 'joint_graph'

    def __init__(self, para, ext_data=None):
        super(AverageArgumentKernelCRF, self).__init__(para, ext_data)
        self.args_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.evm_arg_linear = nn.Linear(self.embedding_dim * 2,
                                        self.embedding_dim)

        if use_cuda:
            self.args_linear.cuda()
            self.evm_arg_linear.cuda()

    def event_embedding(self, mtx_evm, ts_args, mtx_arg_length, ts_arg_mask):
        mtx_p_embedding = self.embedding(mtx_evm)

        if ts_args is None:
            mtx_arg = torch.zeros(mtx_p_embedding.size())
            if use_cuda:
                mtx_arg = mtx_arg.cuda()
        else:
            mtx_arg_embedding_sum = self._argument_sum(ts_args, ts_arg_mask)

            # Remove zero lengths.
            mtx_arg_length[mtx_arg_length == 0] = 1

            broadcast_length = mtx_arg_length.unsqueeze(2).type_as(
                mtx_arg_embedding_sum)
            # Average argument embedding.
            mtx_arg_embedding_aver = mtx_arg_embedding_sum / broadcast_length

            # Non linearly map the argument embeddings.
            mtx_arg = F.tanh(self.args_linear(mtx_arg_embedding_aver))

        mtx_evm_args_cat = torch.cat((mtx_p_embedding, mtx_arg), 2)

        # Non linearly combine event and argument embeddings.
        return F.tanh(self.evm_arg_linear(mtx_evm_args_cat))
