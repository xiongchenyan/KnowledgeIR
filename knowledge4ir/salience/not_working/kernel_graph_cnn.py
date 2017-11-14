import logging

import torch
from torch import nn as nn

from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling
from knowledge4ir.salience.knrm_vote import use_cuda, KNRM


class KernelGraphWalk(SalienceBaseModel):
    """
    no working now... cannot converge. 9/19
    """
    def __init__(self, para, pre_embedding=None):
        super(KernelGraphWalk, self).__init__(para, pre_embedding)
        l_mu, l_sigma = para.form_kernels()
        self.K = len(l_mu)
        self.kp = KernelPooling(l_mu, l_sigma)
        self.embedding = nn.Embedding(para.entity_vocab_size,
                                      para.embedding_dim, padding_idx=0)
        self.layer = para.nb_hidden_layers
        self.l_linear = []
        for __ in xrange(self.layer):
            self.l_linear.append(nn.Linear(self.K, 1, bias=True))
        # self.linear = nn.Linear(self.K, 1, bias=True)
        if pre_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        if use_cuda:
            logging.info('copying parameter to cuda')
            self.embedding.cuda()
            self.kp.cuda()
            # self.linear.cuda()
            for linear in self.l_linear:
                linear.cuda()

        return

    def forward(self, h_packed_data,):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_embedding = self.embedding(mtx_e)
        mtx_embedding = mtx_embedding.div(
            torch.norm(mtx_embedding, p=2, dim=-1, keepdim=True).expand_as(mtx_embedding) + 1e-8
        )

        trans_mtx = torch.matmul(mtx_embedding, mtx_embedding.transpose(-2, -1))
        output = mtx_score
        for linear in self.l_linear:
            kp_mtx = self.kp(trans_mtx, output)
            output = linear(kp_mtx)
            output = output.squeeze(-1)
        if use_cuda:
            return output.cuda()
        else:
            return output


class HighwayKCNN(KNRM):
    def __init__(self, para, pre_embedding=None):
        super(HighwayKCNN, self).__init__(para, pre_embedding)
        self.linear_combine = nn.Linear(2, 1)
        if use_cuda:
            self.linear_combine.cuda()
        return

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        knrm_res = super(HighwayKCNN, self).forward(mtx_e, mtx_score)
        mixed_knrm = torch.cat((knrm_res.unsqueeze(-1), mtx_score.unsqueeze(-1)), -1)
        output = self.linear_combine(mixed_knrm).squeeze(-1)
        return output