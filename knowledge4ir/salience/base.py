"""
utils for salience inference
"""
import json
import logging

import torch
from torch import nn
from torch.autograd import Variable
from traitlets import (
    Int,
    Float,
    List,
    Unicode,
    Bool,
)
import numpy as np
from traitlets.config import Configurable

from knowledge4ir.salience.utils.data_io import DataIO

use_cuda = torch.cuda.is_available()


class NNPara(Configurable):
    embedding_dim = Int(help='embedding dimension').tag(config=True)
    entity_vocab_size = Int(help='total number of entities').tag(config=True)
    event_vocab_size = Int(help='total number of events').tag(config=True)

    nb_hidden_layers = Int(1, help='total number of hidden layers').tag(
        config=True)
    nb_random_walk_steps = Int(1, help='random walk steps').tag(config=True)
    nb_mu = Int(10, help='number of mu').tag(config=True)
    first_k_mu = Int(help='first k mu to use').tag(config=True)
    sigma = Float(0.1, help='sigma').tag(config=True)
    dropout_rate = Float(0, help='dropout rate').tag(config=True)
    train_word_emb = Bool(False, help='whether train word embedding').tag(
        config=True)
    node_feature_dim = Int(0, help='node feature dimension').tag(config=True)
    e_feature_dim = Int(help='entity feature dimension').tag(config=True)
    evm_feature_dim = Int(help='event feature dimension').tag(config=True)
    use_mask = Bool(False, help='whether to use mask').tag(config=True)

    l_hidden_dim = List(Int, default_value=[],
                        help='multi layer DNN hidden dim').tag(config=True)

    desp_sent_len = Int(20,
                        help='the first k words to use in the description').tag(
        config=True)
    kernel_size = Int(3, help='sentence CNN kernel size').tag(config=True)

    l_cnn_length = List(Int, default_value=[1, 2, 3],
                        help='sentence CNN sizes').tag(config=True)
    min_loc_distance = Int(10,
                           help='the minimum distance between two entities '
                                'to receive vote in edge sparse knrm'
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


class ExtData(Configurable):
    """
    external data config and read
    """
    entity_emb_in = Unicode(help='hashed numpy entity embedding path').tag(
        config=True)
    event_emb_in = Unicode(
        help='hashed numpy event embedding path, only used when joint').tag(
        config=True)
    word_emb_in = Unicode(help='hashed numpy word embedding in').tag(
        config=True)
    entity_desp_in = Unicode(help='hashed desp numpy array').tag(config=True)
    entity_rdf_in = Unicode(help='hashed rdf triple numpy array').tag(
        config=True)
    entity_nlss_in = Unicode(
        help='hashed entity natural language support sentence array').tag(
        config=True)

    def __init__(self, **kwargs):
        super(ExtData, self).__init__(**kwargs)
        self.entity_emb = None
        self.word_emb = None
        self.entity_desp = None
        self.entity_rdf = None
        self.entity_nlss = None
        self._load()

    def _load(self):
        if self.entity_emb_in:
            logging.info('loading entity_emb_in [%s]', self.entity_emb_in)
            self.entity_emb = np.load(self.entity_emb_in)
            logging.info('shape %s', json.dumps(self.entity_emb.shape))
        if self.event_emb_in:
            logging.info('loading event_emb_in [%s]', self.event_emb_in)
            self.event_emb = np.load(self.event_emb_in)
        if self.word_emb_in:
            logging.info('loading word_emb_in [%s]', self.word_emb_in)
            self.word_emb = np.load(self.word_emb_in)
            logging.info('shape %s', json.dumps(self.word_emb.shape))
        if self.entity_desp_in:
            logging.info('loading entity_desp_in [%s]', self.entity_desp_in)
            self.entity_desp = np.load(self.entity_desp_in)
            logging.info('shape %s', json.dumps(self.entity_desp.shape))
        if self.entity_rdf_in:
            logging.info('loading entity_rdf_in [%s]', self.entity_rdf_in)
            self.entity_rdf = np.load(self.entity_rdf_in)
            logging.info('shape %s', json.dumps(self.entity_rdf.shape))
        if self.entity_nlss_in:
            logging.info('loading entity_nlss_in [%s]', self.entity_nlss_in)
            self.entity_nlss = np.load(self.entity_nlss_in)
            logging.info('shape %s', json.dumps(self.entity_nlss.shape))
        logging.info('ext data loaded')

    def assert_with_para(self, nn_para):

        if self.entity_emb_in:
            logging.info("Input entity embedding shape is [%d,%d]",
                         self.entity_emb.shape[0], self.entity_emb.shape[1])
            if not nn_para.entity_vocab_size:
                nn_para.entity_vocab_size = self.entity_emb.shape[0]
                nn_para.embedding_dim = self.entity_emb.shape[1]
            else:
                assert nn_para.entity_vocab_size == self.entity_emb.shape[0]
                assert nn_para.embedding_dim == self.entity_emb.shape[1]
        elif self.event_emb_in:
            logging.info("Input event embedding shape is [%d,%d]",
                         self.event_emb.shape[0], self.event_emb.shape[1])
            assert nn_para.event_vocab_size == self.event_emb.shape[0]
            assert nn_para.embedding_dim == self.event_emb.shape[1]
        else:
            logging.warn("Entity embedding not supplied, not asserting.")
            logging.info("Defined entity embedding shape is [%d,%d]",
                         nn_para.entity_vocab_size, nn_para.embedding_dim)


class SalienceBaseModel(nn.Module):
    io_group = 'raw'

    def __init__(self, para, ext_data=None):
        """
        :param para: NNPara
        :param ext_data: external data to use, in ExtData
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
        logging.warn('deprecated, would use the build-in save api of torch')
        return

    def data_io(self, l_lines, io_parser=None):
        if io_parser:
            parser = io_parser
        else:
            parser = DataIO()
        if not parser.l_target_data:
            parser.group_name = self.io_group
            parser.config_target_group()
        return parser.parse_data(l_lines)

    def forward_intermediate(self, h_packed_data):
        return self.forward(h_packed_data)


class KernelPooling(nn.Module):
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
        sum_kernel_value = torch.sum(weighted_kernel_value, dim=-2).clamp(
            min=1e-10)  # add entity freq/weight
        sum_kernel_value = torch.log(sum_kernel_value)
        return sum_kernel_value
