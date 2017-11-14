"""
utils for salience inference
"""
import json
import logging

import torch
from torch import nn, nn as nn
from torch.autograd import Variable
from traitlets import (
    Int,
    Float,
    List,
    Unicode,
    Bool,
)
from traitlets.config import Configurable

from knowledge4ir.salience.kernel_graph_cnn import use_cuda


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


class KernelPooling(nn.Module):
    def __init__(self, l_mu=None, l_sigma=None):
        super(KernelPooling, self).__init__()
        if l_mu is None:
            l_mu = [1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        self.v_mu = Variable(torch.FloatTensor(l_mu), requires_grad=False)
        self.K = len(l_mu)
        if l_sigma is None:
            l_sigma = [1e-3] + [0.1] * (self.v_mu.size()[-1] - 1)
        self.v_sigma = Variable(torch.FloatTensor(l_sigma), requires_grad=False)
        if use_cuda:
            self.v_mu = self.v_mu.cuda()
            self.v_sigma = self.v_sigma.cuda()
        logging.info('[%d] pooling kernels: %s',
                     self.K, json.dumps(zip(l_mu, l_sigma))
                     )
        return

    def forward(self, in_tensor, mtx_score):
        in_tensor = in_tensor.unsqueeze(-1)
        in_tensor = in_tensor.expand(in_tensor.size()[:-1] + (self.K,))
        score = -(in_tensor - self.v_mu) * (in_tensor - self.v_mu)
        kernel_value = torch.exp(score / (2.0 * self.v_sigma * self.v_sigma))
        mtx_score = mtx_score.unsqueeze(-1).unsqueeze(1)
        mtx_score = mtx_score.expand_as(kernel_value)
        weighted_kernel_value = kernel_value * mtx_score
        sum_kernel_value = torch.sum(weighted_kernel_value, dim=-2).clamp(min=1e-10)  # add entity freq/weight
        sum_kernel_value = torch.log(sum_kernel_value)
        return sum_kernel_value