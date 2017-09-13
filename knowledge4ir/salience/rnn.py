"""
rnn model
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging
use_cuda = torch.cuda.is_available()


class BiGRU(nn.Module):
    """
    input: list of document entities + frequency, target e id
    output: p(target e id is salient | not salient)
    """

    def __init__(self, layer, vocab_size, embedding_dim, pre_embedding=None):
        super(BiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(1, 2, bias=True)
        if use_cuda:
            logging.info('copying parameter to cuda')
            self.embedding.cuda()
            self.linear.cuda()
        self.layer = layer
        if pre_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        return

    def forward(self, v_e, v_score):
        """
        return probability of each one being salient
        :param v_e: the input entity id's, has to be Variable()
        :param v_score: the initial weights on each entity, has to be Variable()
        :return: score for each one
        """
        return

