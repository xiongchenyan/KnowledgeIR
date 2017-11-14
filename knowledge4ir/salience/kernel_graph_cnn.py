"""
kernel pooling layer
init:
    v_mu: a 1-d dimension of mu's
    sigma: the sigma
input:
    similar to Linear()
        a n-D tensor, last dimension is the one to enforce kernel pooling
output:
    n-K tensor, K is the v_mu.size(), number of kernels
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling

use_cuda = torch.cuda.is_available()


class KernelGraphCNN(SalienceBaseModel):

    def __init__(self, para, pre_embedding=None):
        super(KernelGraphCNN, self).__init__(para, pre_embedding)
        l_mu, l_sigma = para.form_kernels()
        self.K = len(l_mu)
        self.kp = KernelPooling(l_mu, l_sigma)
        self.embedding = nn.Embedding(para.entity_vocab_size,
                                      para.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=para.dropout_rate)
        self.linear = nn.Linear(self.K, 1, bias=True)
        if pre_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        if use_cuda:
            logging.info('copying parameter to cuda')
            self.embedding.cuda()
            self.kp.cuda()
            self.linear.cuda()
        self.layer = para.nb_hidden_layers
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
        trans_mtx = self.dropout(trans_mtx)
        kp_mtx = self.kp(trans_mtx, mtx_score)
        output = self.linear(kp_mtx)
        output = output.squeeze(-1)
        return output

    def save_model(self, output_name):
        logging.info('saving knrm embedding and linear weights to [%s]', output_name)
        emb_mtx = self.embedding.weight.data.cpu().numpy()
        np.save(open(output_name + '.emb.npy', 'w'), emb_mtx)
        np.save(open(output_name + '.linear.npy', 'w'),
                self.linear.weight.data.cpu().numpy())


