"""
k-nrm vote from both entities and words
entity vote kernels
word vote kernels
linear combination
"""


import logging
import numpy as np
import torch
import torch.nn as nn
import json
from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling
from knowledge4ir.salience.knrm_vote import KNRM

use_cuda = torch.cuda.is_available()


class DuetKNRM(KNRM):
    def __init__(self, para, ext_data=None):
        super(DuetKNRM, self).__init__(para, ext_data)
        assert ext_data.word_emb is not None
        self.word_embedding = nn.Embedding(
            ext_data.word_emb.shape[0],
            ext_data.word_emb.shape[1],
            padding_idx=0,
        )
        logging.info('init word embedding with pre-trained [%s]', ext_data.word_emb_in)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(ext_data.word_emb)
        )
        self.duet_linear = nn.Linear(self.K * 2, 1, bias=True)
        if use_cuda:
            logging.info('copying duet knrm parameters to cuda')
            self.word_embedding.cuda()
            self.duet_linear.cuda()
        return

    def forward(self, h_packed_data):
        """
        knrm from e
        knrm from w
        linear combine the kernel
        :param h_packed_data:
        :return:
        """
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        assert 'mtx_w' in h_packed_data   # has word sequence in it
        assert 'mtx_w_score' in h_packed_data

        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_w = h_packed_data['mtx_w']
        mtx_w_score = h_packed_data['mtx_w_score']

        e_emb = self.embedding(mtx_e)
        w_emb = self.embedding(mtx_w)

        entity_vote_kernels = self._kernel_scores(e_emb, mtx_score)
        word_vote_kernels = self._kernel_vote(e_emb, w_emb, mtx_w_score)

        fuse_vote = torch.cat([entity_vote_kernels, word_vote_kernels], dim=-1)
        output = self.duet_linear(fuse_vote).squeeze(-1)
        return output

    def _kernel_vote(self, e_emb, w_emb, w_score):
        e_emb = self._normalize(e_emb)
        w_emb = self._normalize(w_emb)

        trans_mtx = torch.matmul(e_emb, w_emb.transpose(-2, -1))
        trans_mtx = self.dropout(trans_mtx)
        return self.kp(trans_mtx, w_score)

    def _normalize(self, mtx_embedding):
        return nn.functional.normalize(mtx_embedding, p=2, dim=-1)
        # logging.info('normalize shape %s', json.dumps(mtx_embedding.size()))
        # z = torch.norm(mtx_embedding, p=2, dim=-1, keepdim=True).expand_as(mtx_embedding) + 1e-8
        # logging.info('z shape %s', json.dumps(z.size()))
        # mtx_embedding = mtx_embedding.div(z)
        #
        # return mtx_embedding
