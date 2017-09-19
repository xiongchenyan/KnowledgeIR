"""
utils for salience inference
"""

from ranking_loss import *

from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    List
)
from torch import nn


class NNPara(Configurable):
    embedding_dim = Int(help='embedding dimension').tag(config=True)
    entity_vocab_size = Int(help='total number of entities').tag(config=True)
    nb_hidden_layers = Int(1, help='total number of hidden layers').tag(config=True)
    nb_random_walk_steps = Int(1, help='random walk steps').tag(config=True)
    nb_mu = Int(10, help='number of mu').tag(config=True)
    sigma = Float(0.1, help='sigma').tag(config=True)

    def form_kernels(self):
        l_mu = [1.0]
        l_sigma = [1e-3]
        bin_range = 2.0 / self.nb_mu
        st = 1.0 - bin_range / 2.0
        for i in range(self.nb_mu):
            l_mu.append(st - i * bin_range)
            l_sigma.append(self.sigma)
        return l_mu, l_sigma


class SalienceBaseModel(nn.Module):

    def __init__(self, para, pre_emb=None):
        """
        para is NNPara
        pre_emb is a numpy matrix of entity embeddings
        :param para:
        :param pre_emb:
        """
        super(SalienceBaseModel, self).__init__()

    def forward(self, mtx_e, mtx_score):
        """

        :param mtx_e: batch * e per doc, entity ids
        :param mtx_score: batch * e per doc, pre-given entity scores, typically frequency
        :return:
        """
