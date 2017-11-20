"""
11/20/2017 Chenyan

incorporate the natural language support sentences (NLSS) for the entities
    NLSS's now come from sentences in the abstract

two models:
    sentence CNN + n-gram max pooling + sentence level max pooling
    use attention on the sentence's CNN embedding
        1) start with attention with the entity's embedding first... Intuition not quit clear
        2) will need better attention in the future, should be some local attention
            2.a) compare with local content?
"""


import logging
import numpy as np
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling
from knowledge4ir.salience.knrm_vote import KNRM

use_cuda = torch.cuda.is_available()


class NlssCnnKnrm(KNRM):

    def __init__(self, para, ext_data=None):
        super(NlssCnnKnrm, self).__init__(para, ext_data)

        assert ext_data.word_emb is not None
        assert ext_data.entity_nlss is not None
        assert para.kernel_size

        # entity * nlss * words
        self.e_nlss = Variable(torch.LongTensor(ext_data.entity_nlss))
        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))

        self.sentence_cnn = torch.nn.Conv1d(
            in_channels=para.embedding_dim,
            out_channels=para.embedding_dim,
            kernel_size=para.kernel_size,
            bias=False,
        )
        self.emb_merge = nn.Linear(
            para.embedding_dim * 2,
            para.embedding_dim,
            bias=False
        )
        if use_cuda:
            self.sentence_cnn.cuda()
            self.word_emb.cuda()
            self.e_nlss = self.e_nlss.cuda()
            self.emb_merge.cuda()

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_embedding = self.embedding(mtx_e)    # memory based embedding

        ts_nlss = self.e_nlss[mtx_e.view(-1)].view(
            mtx_e.size() + self.e_nlss.size()[-2:]
        )     # batch, e id, nlss, words

        v_nlss_words = ts_nlss.view(-1)
        ts_nlss_emb = self.word_emb(v_nlss_words)

        # batch * entity * nlss *  words * word embedding
        ts_nlss_emb = ts_nlss_emb.view(ts_nlss.size() + ts_nlss_emb.size()[-1:])

        # reshape for CNN:
        # now is (batch * entity * nlss) * nlss's words * word embedding
        ts_nlss_emb = ts_nlss_emb.view((-1,) + ts_nlss_emb.size()[-2:])
        ts_nlss_emb = ts_nlss_emb.transpose(-1, -2)   # now batch * embedding * words
        logging.debug('cnn input sequence shape %s', json.dumps(ts_nlss_emb.size()))
        cnn_filter = self.sentence_cnn(ts_nlss_emb)
        logging.debug('cnn raw output sequence shape %s', json.dumps(ts_nlss_emb.size()))
        cnn_filter = cnn_filter.transpose(-2, -1).contiguous()   # batch * strides * filters
        cnn_filter = cnn_filter.view(
            ts_nlss.size()[:-1] + cnn_filter.size()[-2:]
        )    # batch * entity * nlss * strides * filters
        logging.debug('cnn out converted to shape %s', json.dumps(cnn_filter.size()))
        cnn_emb, __ = torch.max(
            cnn_filter, dim=-2, keepdim=False
        )
        cnn_emb, __ = torch.max(
            cnn_emb, dim=-2, keepdim=False
        )
        logging.debug('max pooled CNN Emb shape %s', json.dumps(cnn_emb.size()))
        enriched_e_embedding = self.emb_merge(
            torch.cat((mtx_embedding, cnn_emb), dim=-1)
        )

        return self._knrm_opt(enriched_e_embedding, mtx_score)


