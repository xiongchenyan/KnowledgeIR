"""
only allow vote between entities within a certain distance
those appear further away will not vote for each other
minimum distance set by para.min_loc_distance
"""


import logging
import numpy as np
import torch
import torch.nn as nn
import json

from torch import nn as nn
from knowledge4ir.salience.knrm_vote import KNRM

use_cuda = torch.cuda.is_available()


class AdjKNRM(KNRM):
    def __init__(self, para, ext_data=None):
        super(AdjKNRM, self).__init__(para, ext_data)
        self.min_distance = para.min_loc_distance

    def forward(self, h_packed_data):
        assert 'ts_distance' in h_packed_data
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        ts_distance = h_packed_data['ts_distance']
        ts_distance = (ts_distance <= self.min_distance) * (ts_distance >= 0)
        ts_distance = ts_distance.float()

        mtx_embedding = self.embedding(mtx_e)
        kp_mtx = self._sparse_kernel_vote(mtx_embedding, mtx_embedding, mtx_score, ts_distance)
        output = self.linear(kp_mtx)
        output = output.squeeze(-1)
        return output

    def _sparse_kernel_vote(self, target_emb, voter_emb, voter_score, target_vote_adj_mtx):
        """
        only receive vote if there is an edge in between the target and voter
        :param target_emb:
        :param voter_emb:
        :param voter_score:
        :param target_vote_adj_mtx: Float Type
        :return:
        """
        target_emb = nn.functional.normalize(target_emb, p=2, dim=-1)
        voter_emb = nn.functional.normalize(voter_emb, p=2, dim=-1)

        trans_mtx = torch.matmul(target_emb, voter_emb.transpose(-2, -1))
        trans_mtx = trans_mtx * target_vote_adj_mtx
        trans_mtx = self.dropout(trans_mtx)
        return self.kp(trans_mtx, voter_score)

