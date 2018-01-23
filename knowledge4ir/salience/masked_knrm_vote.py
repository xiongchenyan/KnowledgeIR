"""
kernel based votes from other entities
"""

import logging

import numpy as np
import torch
from torch import nn as nn

from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling

use_cuda = torch.cuda.is_available()


class MaskKNRM(SalienceBaseModel):
    def __init__(self, para, ext_data=None):
        super(MaskKNRM, self).__init__(para, ext_data)
        l_mu, l_sigma = para.form_kernels()
        self.K = len(l_mu)
        self.kp = KernelPooling(l_mu, l_sigma)
        self.dropout = nn.Dropout(p=para.dropout_rate)
        self.linear = nn.Linear(self._feature_size(), 1, bias=True)
        self._load_embedding(para, ext_data)
        if use_cuda:
            logging.info('copying knrm parameter to cuda')
            self.embedding.cuda()
            self.kp.cuda()
            self.linear.cuda()
        self.layer = para.nb_hidden_layers
        return

    def _feature_size(self):
        raise NotImplementedError

    def _load_embedding(self, para, ext_data):
        self.embedding = nn.Embedding(para.entity_vocab_size,
                                      para.embedding_dim, padding_idx=0)
        if ext_data.entity_emb is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(ext_data.entity_emb))

    def forward(self, h_packed_data):
        raise NotImplementedError

    def _masked_kernel_scores(self, mask, mtx_embedding, mtx_score):
        masked_embedding = mtx_embedding * mask.unsqueeze(-1)
        return self.__kernel_vote(masked_embedding, masked_embedding, mtx_score)

    def _kernel_scores(self, mtx_embedding, mtx_score):
        return self.__kernel_vote(mtx_embedding, mtx_embedding, mtx_score)

    def _masked_kernel_vote(self, target_emb, target_mask, voter_emb,
                            voter_mask, voter_score):
        masked_target = target_emb * target_mask.unsqueeze(-1)
        masked_voter = voter_emb * voter_mask.unsqueeze(-1)
        return self.__kernel_vote(masked_target, masked_voter, voter_score)

    def __kernel_vote(self, target_emb, voter_emb, voter_score):
        target_emb = nn.functional.normalize(target_emb, p=2, dim=-1)
        voter_emb = nn.functional.normalize(voter_emb, p=2, dim=-1)

        trans_mtx = torch.matmul(target_emb, voter_emb.transpose(-2, -1))
        trans_mtx = self.dropout(trans_mtx)
        return self.kp(trans_mtx, voter_score)

    def _forward_kernel_with_features(self, mtx_embedding,
                                      mtx_score, node_features):
        kp_mtx = self.__kernel_vote(mtx_embedding, mtx_embedding, mtx_score)
        # Combine with node features.
        features = torch.cat((kp_mtx, node_features), -1)
        output = self.linear(features).squeeze(-1)
        return output

    def _forward_kernel_with_mask_and_features(self, mask, mtx_embedding,
                                               mtx_score, node_features):
        kp_mtx = self._masked_kernel_scores(mask, mtx_embedding, mtx_score)
        # Combine with node features.
        features = torch.cat((kp_mtx, node_features), -1)
        output = self.linear(features).squeeze(-1)
        return output
