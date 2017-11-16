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


import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling
from knowledge4ir.salience.knrm_vote import KNRM

use_cuda = torch.cuda.is_available()


class DespWordAvgEmbKNRM(KNRM):
    """
    attention weighted word average as entity additional embedding
        word embedding + attention weights -> entity's additional embedding
    """

    def __init__(self, para, ext_data=None):
        super(DespWordAvgEmbKNRM, self).__init__(para, ext_data)
        assert ext_data.word_emb
        assert ext_data.entity_desp

        self.e_att_embedding = nn.Embedding(para.entity_vocab_size,
                                            para.embedding_dim, padding_idx=0)
        if ext_data.entity_emb is not None:
            self.e_att_embedding.weight.data.copy_(torch.from_numpy(ext_data.entity_emb))

        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data._copy(torch.from_numpy(ext_data.word_emb))

        self.word_att_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                         ext_data.word_emb.shape[1], padding_idx=0)
        self.word_att_emb.weight.data._copy(torch.from_numpy(ext_data.word_emb))

        self.emb_merge = torch.linear(para.embedding_dim + ext_data.word_emb.shape[1], para.embedding_dim, bias=False)

        self.e_desp_mtx = torch.LongTensor(ext_data.entity_desp)

        if use_cuda:
            self.word_att_emb.cuda()
            self.word_emb.cuda()
            self.e_desp_mtx.cuda()
            self.e_att_embedding.cuda()
            self.emb_merge.cuda()

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_embedding = self.embedding(mtx_e)    # memory based embedding
        mtx_att_embedding = self.e_att_embedding(mtx_e)

        ts_desp_att_emb, ts_desp_content_emb = self._form_ext_desp(mtx_e)

        att_word_emb = self._att_avg_emb(mtx_att_embedding, ts_desp_att_emb, ts_desp_content_emb)

        enriched_e_embedding = self.emb_merge(
            torch.cat((mtx_embedding, att_word_emb), dim=-1)
        )

        return self._knrm_opt(enriched_e_embedding, mtx_score)

    def _form_ext_desp(self, mtx_e):
        ts_desp = self.e_desp_mtx[mtx_e.view(-1)].view(
            mtx_e.size() + (self.e_desp_mtx.size()[-1],)
        )     # batch, e id, desp word id

        ts_desp_content_emb = self.word_emb(Variable(ts_desp))
        ts_desp_att_emb = self.word_att_emb(Variable(ts_desp))     # batch, e id, desp word id, emb
        return ts_desp_att_emb, ts_desp_content_emb

    def _att_avg_emb(self, mtx_att_embedding, ts_desp_att_emb, ts_desp_content_emb):
        att_score = torch.matmul(
            ts_desp_att_emb, mtx_att_embedding.unsqueeze(-1)).squeeze(-1)    # batch, e id, desp word' weights
        att_score = nn.functional.softmax(att_score, dim=-1)
        att_word_emb = torch.matmul(
            att_score.unsqueeze(-2), ts_desp_content_emb).unsqueeze(-2)   # avg desp word emb for each entity
        return att_word_emb


class DespSentRNNEmbedKNRM(KNRM):
    """
    rnn of the description's first 10 words
    """

    def __init__(self, para, ext_data=None):
        super(DespSentRNNEmbedKNRM, self).__init__(para, ext_data)

        assert ext_data.word_emb
        assert ext_data.entity_desp
        assert para.desp_sent_len
        self.e_desp_mtx = self.e_desp_mtx[:, :para.desp_sent_len]

        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data._copy(torch.from_numpy(ext_data.word_emb))

        self.desp_rnn = torch.nn.GRU(
            input_size=para.embedding_dim,
            hidden_size=para.embedding_dim,
            num_layers=1,
            bias=False,
            batch_first=True,
            bidirectional=True
        )
        self.emb_merge = nn.linear(
            para.embedding_dim * 3,
            para.embedding_dim,
            bias=False
        )
        if use_cuda:
            self.desp_rnn.cuda()
            self.word_emb.cuda()

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_embedding = self.embedding(mtx_e)    # memory based embedding

        ts_desp = self.e_desp_mtx[mtx_e.view(-1)].view(
            mtx_e.size() + (self.e_desp_mtx.size()[-1],)
        )     # batch, e id, desp word id

        ts_desp_emb = self.word_emb(Variable(ts_desp))     # batch, e, word, embedding
        ts_desp_emb = ts_desp_emb.view((-1,) + ts_desp_emb.size()[-2:])

        h0 = Variable(torch.randn(2, ts_desp_emb.size()[-2], ts_desp_emb.size()[-1]))

        __, desp_rnn_out = self.desp_rnn(ts_desp_emb, h0)

        desp_rnn_out.transpose(0, 1)
        forward_rnn_out = desp_rnn_out[:, 0, :].contiguous()  # now batch-embedding
        backward_rnn_out = desp_rnn_out[:, 1, :].contiguous()

        forward_rnn_out = forward_rnn_out.view(mtx_e.size() + (-1,))
        backward_rnn_out = backward_rnn_out.view_as(forward_rnn_out)

        enriched_e_embedding = self.emb_merge(
            torch.cat((mtx_embedding, forward_rnn_out, backward_rnn_out), dim=-1)
        )

        return self._knrm_opt(enriched_e_embedding, mtx_score)
