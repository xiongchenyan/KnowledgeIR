"""
utils for salience inference
"""

from torch import nn
from traitlets import (
    Int,
    Float,
    List,
    Unicode,
    Bool,
)
from traitlets.config import Configurable


class NNPara(Configurable):
    embedding_dim = Int(help='embedding dimension').tag(config=True)
    entity_vocab_size = Int(help='total number of entities').tag(config=True)
    nb_hidden_layers = Int(1, help='total number of hidden layers').tag(config=True)
    nb_random_walk_steps = Int(1, help='random walk steps').tag(config=True)
    nb_mu = Int(10, help='number of mu').tag(config=True)
    first_k_mu = Int(help='first k mu to use').tag(config=True)
    sigma = Float(0.1, help='sigma').tag(config=True)
    dropout_rate = Float(0, help='dropout rate').tag(config=True)
    train_word_emb = Bool(False, help='whether train word embedding').tag(config=True)
    node_feature_dim = Int(10, help='node feature dimension').tag(config=True)
    l_hidden_dim = List(Int, default_value=[], help='multi layer DNN hidden dim').tag(config=True)
    word_emb_in = Unicode(
        help='pre trained word embedding, npy format, must be comparable with entity embedding'
    ).tag(config=True)

    def form_kernels(self):
        l_mu = [1.0]
        l_sigma = [1e-3]
        bin_range = 2.0 / self.nb_mu
        st = 1.0 - bin_range / 2.0
        for i in range(self.nb_mu):
            l_mu.append(st - i * bin_range)
            l_sigma.append(self.sigma)
        if self.first_k_mu:
            l_mu = l_mu[:self.first_k_mu]
            l_sigma = l_sigma[:self.first_k_mu]
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

    def forward(self, h_packed_data):
        """
        :param h_packed_data: the packed data to get, can contain:
            mtx_e: batch * e per doc, entity ids
            mtx_score: batch * e per doc, pre-given entity scores, typically frequency
            ts_feature: one feature vector for each e
        :return:
        """

    def save_model(self, output_name):
        return