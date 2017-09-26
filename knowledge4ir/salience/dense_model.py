"""
lr on entity embedding
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
from knowledge4ir.salience.utils import SalienceBaseModel
use_cuda = torch.cuda.is_available()


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


class FeatureLR(SalienceBaseModel):

    def __init__(self, para, pre_embedding=None):
        super(FeatureLR, self).__init__(para, pre_embedding)
        self.node_feature_dim = para.node_feature_dim
        self.l_hidden_dim = para.l_hidden_dim
        last_dim = self.node_feature_dim
        self.l_node_lr = []
        for hidden_d in self.l_hidden_dim:
            self.l_node_lr.append(nn.Linear(last_dim, hidden_d, bias=False))
            last_dim = hidden_d
        self.l_node_lr.append(nn.Linear(last_dim, 1, bias=False))

        if use_cuda:
            for linear in self.l_node_lr:
                linear.cuda()

    def forward(self, h_packed_data):
        assert 'ts_feature' in h_packed_data
        ts_feature = h_packed_data['ts_feature']
        middle = ts_feature
        for linear in self.l_node_lr:
            middle = F.tanh(linear(middle))
        output = middle.squeeze(-1)
        return output

    def save_model(self, output_name):
        logging.info('saving node feature linear weights to [%s]', output_name)
        for p in xrange(len(self.l_node_lr)):
            np.save(open(output_name + '.linear_%d.npy' % p, 'w'),
                    self.l_node_lr[p].weight.data.cpu().numpy())
