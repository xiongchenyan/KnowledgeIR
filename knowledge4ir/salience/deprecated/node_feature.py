import logging

import torch
from torch import nn as nn

from knowledge4ir.salience.baseline.node_feature import use_cuda


class EmbeddingLR(nn.Module):
    """
    input: matrix's |doc||v_e|, |doc||v_score| of these v_e
        e=-1 is padding
    output: p(target e id is salient)
    """

    def __init__(self, layer, vocab_size, embedding_dim, pre_embedding=None):
        super(EmbeddingLR, self).__init__()
        self.layer = layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.l_linear = []
        for p in xrange(layer):
            out_dim = embedding_dim if p < layer - 1 else 1
            self.l_linear.append(nn.Linear(embedding_dim, out_dim, bias=False))
        if pre_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        if use_cuda:
            logging.info('copying parameter to cuda')
            self.embedding.cuda()
            for linear in self.l_linear:
                linear.cuda()
        return

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_embedding = self.embedding(mtx_e)
        output = mtx_embedding
        for linear in self.l_linear:
            output = linear(output)

        output = output.squeeze(-1)
        return output