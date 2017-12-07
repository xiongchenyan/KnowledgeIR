import logging

import torch
from torch import nn as nn

from knowledge4ir.salience.duet_knrm import use_cuda
from knowledge4ir.salience.knrm_vote import KNRM


class WordKNRM(KNRM):
    def __init__(self, para, ext_data=None):
        super(WordKNRM, self).__init__(para, ext_data)
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
        self.duet_linear = nn.Linear(self.K, 1, bias=True)
        if use_cuda:
            logging.info('copying word knrm parameters to cuda')
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
        # assert 'mtx_score' in h_packed_data
        assert 'mtx_w' in h_packed_data   # has word sequence in it
        assert 'mtx_w_score' in h_packed_data

        mtx_e = h_packed_data['mtx_e']
        # mtx_score = h_packed_data['mtx_score']
        mtx_w = h_packed_data['mtx_w']
        mtx_w_score = h_packed_data['mtx_w_score']

        e_emb = self.embedding(mtx_e)
        w_emb = self.word_embedding(mtx_w)

        # entity_vote_kernels = self._kernel_scores(e_emb, mtx_score)
        word_vote_kernels = self._kernel_vote(e_emb, w_emb, mtx_w_score)

        # fuse_vote = torch.cat([entity_vote_kernels, word_vote_kernels], dim=-1)
        output = self.duet_linear(word_vote_kernels).squeeze(-1)
        return output

    def _kernel_vote(self, target_emb, voter_emb, voter_score):
        target_emb = nn.functional.normalize(target_emb, p=2, dim=-1)
        voter_emb = nn.functional.normalize(voter_emb, p=2, dim=-1)

        trans_mtx = torch.matmul(target_emb, voter_emb.transpose(-2, -1))
        trans_mtx = self.dropout(trans_mtx)
        return self.kp(trans_mtx, voter_score)