"""
kernel based votes from other entities
"""

import logging

import numpy as np
import torch
from torch import nn as nn

from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling

use_cuda = torch.cuda.is_available()


class KNRM(SalienceBaseModel):
    def __init__(self, para, ext_data=None):
        super(KNRM, self).__init__(para, ext_data)
        l_mu, l_sigma = para.form_kernels()
        self.K = len(l_mu)
        self.kp = KernelPooling(l_mu, l_sigma)
        self.dropout = nn.Dropout(p=para.dropout_rate)
        self.linear = nn.Linear(self.K, 1, bias=True)
        self._load_embedding(para, ext_data)
        if use_cuda:
            logging.info('copying knrm parameter to cuda')
            self.embedding.cuda()
            self.kp.cuda()
            self.linear.cuda()
        self.layer = para.nb_hidden_layers
        return

    def _load_embedding(self, para, ext_data):
        self.embedding = nn.Embedding(para.entity_vocab_size,
                                      para.embedding_dim, padding_idx=0)
        if ext_data.entity_emb is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(ext_data.entity_emb))

    def forward(self, h_packed_data):
        kp_mtx = self._forward_to_kernels(h_packed_data)
        output = self.linear(kp_mtx)
        output = output.squeeze(-1)
        return output

    # def save_model(self, output_name):
    #     """
    #     to be deprecated, will use Torch's general model save/load API
    #     :param output_name:
    #     :return:
    #     """
    #     logging.info('saving knrm embedding and linear weights to [%s]',
    #                  output_name)
    #     emb_mtx = self.embedding.weight.data.cpu().numpy()
    #     np.save(open(output_name + '.emb.npy', 'w'), emb_mtx)
    #     np.save(open(output_name + '.linear.npy', 'w'),
    #             self.linear.weight.data.cpu().numpy())

    def _knrm_opt(self, mtx_embedding, mtx_score):
        kp_mtx = self._kernel_scores(mtx_embedding, mtx_score)
        output = self.linear(kp_mtx)
        output = output.squeeze(-1)
        return output

    def _kernel_scores(self, mtx_embedding, mtx_score):
        return self._kernel_vote(mtx_embedding, mtx_embedding, mtx_score)

    def _kernel_vote(self, target_emb, voter_emb, voter_score):
        target_emb = nn.functional.normalize(target_emb, p=2, dim=-1)
        voter_emb = nn.functional.normalize(voter_emb, p=2, dim=-1)

        trans_mtx = torch.matmul(target_emb, voter_emb.transpose(-2, -1))
        trans_mtx = self.dropout(trans_mtx)
        return self.kp(trans_mtx, voter_score)

    def _forward_to_kernels(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_embedding = self.embedding(mtx_e)
        kp_mtx = self._kernel_scores(mtx_embedding, mtx_score)
        return kp_mtx

    def forward_intermediate(self, h_packed_data):
        return self._forward_to_kernels(h_packed_data)
