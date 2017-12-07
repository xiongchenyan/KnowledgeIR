"""
11/15/2017 Chenyan

models to use external semantics in entity salience modeling

external semantics now:
    description in ExtData
    will have NLSS and RDF in the future

model:
    get an additional entity embedding from the ext semantics
    embedding can be:
        average of desp's first 10 words
        weighted average of the first 10 words
        desp RNN
        desp's senence's RNN with attention
        desp CNN
    to use the additional embedding
        add | concatenate with e's learned embedding | caocatenate + projection to a new embedding

    cost:
        for each doc, need get all its e's external semantics

"""


import json
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from knowledge4ir.salience.knrm_vote import KNRM

use_cuda = torch.cuda.is_available()


class GlossCNNKNRM(KNRM):
    """
    Cnn of the description's first 20 words
    multiple CNN filters
    """

    def __init__(self, para, ext_data=None):
        super(GlossCNNKNRM, self).__init__(para, ext_data)

        assert ext_data.word_emb is not None
        assert ext_data.entity_desp is not None
        assert para.desp_sent_len
        self.e_desp_mtx = Variable(torch.LongTensor(ext_data.entity_desp[:, :para.desp_sent_len]))
        # self.e_desp_mtx = self.e_desp_mtx[:, :para.desp_sent_len]
        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))
        self.l_gloss_cnn = []
        for k_size in para.l_cnn_length:
            self.l_gloss_cnn.append(torch.nn.Conv1d(
                in_channels=para.embedding_dim,
                out_channels=para.embedding_dim,
                kernel_size=k_size,
                bias=False,
            ))
        self.emb_merge = nn.Linear(
            para.embedding_dim * (len(para.l_cnn_length) + 1),
            para.embedding_dim,
            bias=False
        )
        if use_cuda:
            for i in xrange(len(self.l_gloss_cnn)):
                self.l_gloss_cnn[i].cuda()
            self.word_emb.cuda()
            self.e_desp_mtx = self.e_desp_mtx.cuda()
            self.emb_merge.cuda()
            self.word_emb.cuda()
            self.e_desp_mtx = self.e_desp_mtx.cuda()
            self.emb_merge.cuda()

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_embedding = self.embedding(mtx_e)    # memory based embedding

        ts_desp = self.e_desp_mtx[mtx_e.view(-1)].view(
            mtx_e.size() + (self.e_desp_mtx.size()[-1],)
        )     # batch, e id, desp word id

        v_desp_words = ts_desp.view(-1)
        ts_desp_emb = self.word_emb(v_desp_words)

        # batch * entity * desp words * word embedding
        ts_desp_emb = ts_desp_emb.view(ts_desp.size() + ts_desp_emb.size()[-1:])

        l_cnn_emb = []
        for cnn in self.l_gloss_cnn:
            l_cnn_emb.append(self._sentence_cnn(ts_desp_emb, mtx_e, cnn))

        enriched_e_embedding = self.emb_merge(
            torch.cat([mtx_embedding] + l_cnn_emb, dim=-1)
        )

        return self._knrm_opt(enriched_e_embedding, mtx_score)

    def _sentence_cnn(self, ts_desp_emb, mtx_e, cnn):
        ts_desp_emb = ts_desp_emb.view((-1,) + ts_desp_emb.size()[-2:])
        ts_desp_emb = ts_desp_emb.transpose(-1, -2)   # now batch * embedding * words
        logging.debug('cnn input sequence shape %s', json.dumps(ts_desp_emb.size()))
        cnn_filter = cnn(ts_desp_emb)
        logging.debug('cnn raw output sequence shape %s', json.dumps(ts_desp_emb.size()))
        cnn_filter = cnn_filter.transpose(-2, -1).contiguous()   # batch * strides * filters
        cnn_filter = cnn_filter.view(
            mtx_e.size() + cnn_filter.size()[-2:]
        )    # batch * entity * strides * filters
        logging.debug('cnn out converted to shape %s', json.dumps(cnn_filter.size()))
        cnn_emb, __ = torch.max(
            cnn_filter, dim=-2, keepdim=False
        )
        logging.debug('max pooled CNN Emb shape %s', json.dumps(cnn_emb.size()))
        return cnn_emb
