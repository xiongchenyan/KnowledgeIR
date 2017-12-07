"""
k-nrm vote from both entities and words
entity vote kernels
word vote kernels
linear combination
"""


import json
import logging

import torch
from torch import nn as nn
from torch.autograd import Variable

from knowledge4ir.salience.external_semantics.description import use_cuda
from knowledge4ir.salience.knrm_vote import KNRM

use_cuda = torch.cuda.is_available()


class DuetKNRM(KNRM):
    def __init__(self, para, ext_data=None):
        super(DuetKNRM, self).__init__(para, ext_data)
        assert ext_data.word_emb is not None
        self.word_embedding = nn.Embedding(
            ext_data.word_emb.shape[0],
            ext_data.word_emb.shape[1],
            padding_idx=0,
        )
        logging.info('init word embedding with pre-trained [%s]', ext_data.word_emb_in)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(ext_data.word_emb)
        )
        self.duet_linear = nn.Linear(self.K * 2, 1, bias=True)
        if use_cuda:
            logging.info('copying duet knrm parameters to cuda')
            self.word_embedding.cuda()
            self.duet_linear.cuda()
        return

    def forward(self, h_packed_data):
        """
        knrm from e
        knrm from w
        linear combine the kernel
        :param h_packed_data:
        :return:
        """
        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        assert 'mtx_w' in h_packed_data   # has word sequence in it
        assert 'mtx_w_score' in h_packed_data

        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_w = h_packed_data['mtx_w']
        mtx_w_score = h_packed_data['mtx_w_score']

        e_emb = self.embedding(mtx_e)
        w_emb = self.word_embedding(mtx_w)

        entity_vote_kernels = self._kernel_scores(e_emb, mtx_score)
        word_vote_kernels = self._kernel_vote(e_emb, w_emb, mtx_w_score)

        fuse_vote = torch.cat([entity_vote_kernels, word_vote_kernels], dim=-1)
        output = self.duet_linear(fuse_vote).squeeze(-1)
        return output

    def _kernel_vote(self, e_emb, w_emb, w_score):
        e_emb = nn.functional.normalize(e_emb, p=2, dim=-1)
        w_emb = nn.functional.normalize(w_emb, p=2, dim=-1)

        trans_mtx = torch.matmul(e_emb, w_emb.transpose(-2, -1))
        trans_mtx = self.dropout(trans_mtx)
        return self.kp(trans_mtx, w_score)

    # def _normalize(self, mtx_embedding):
    #     return nn.functional.normalize(mtx_embedding, p=2, dim=-1)
        # logging.info('normalize shape %s', json.dumps(mtx_embedding.size()))
        # z = torch.norm(mtx_embedding, p=2, dim=-1, keepdim=True).expand_as(mtx_embedding) + 1e-8
        # logging.info('z shape %s', json.dumps(z.size()))
        # mtx_embedding = mtx_embedding.div(z)
        #
        # return mtx_embedding


class DuetGlossCNN(DuetKNRM):
    def __init__(self, para, ext_data=None):
        super(DuetGlossCNN, self).__init__(para, ext_data)

        assert ext_data.word_emb is not None
        assert ext_data.entity_desp is not None
        assert para.desp_sent_len
        self.e_desp_mtx = Variable(torch.LongTensor(ext_data.entity_desp[:, :para.desp_sent_len]))
        # self.e_desp_mtx = self.e_desp_mtx[:, :para.desp_sent_len]
        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))
        self.l_gloss_cnn = []
        self.l_gloss_linear = []
        for k_size in para.l_cnn_length:
            self.l_gloss_cnn.append(torch.nn.Conv1d(
                in_channels=para.embedding_dim,
                out_channels=para.embedding_dim,
                kernel_size=k_size,
                bias=False,
            ))
            self.l_gloss_linear.append(nn.Linear(self.K, 1, bias=True))

        self.emb_merge = nn.Linear(
            para.embedding_dim * 2,
            para.embedding_dim,
            bias=False
        )
        if use_cuda:
            for i in xrange(len(self.l_gloss_cnn)):
                self.l_gloss_cnn[i].cuda()
                self.l_gloss_linear[i].cuda()
            self.word_emb.cuda()
            self.e_desp_mtx = self.e_desp_mtx.cuda()
            self.emb_merge.cuda()

    def forward(self, h_packed_data):
        duet_knrm_score = super(DuetGlossCNN, self).forward(h_packed_data)

        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        assert 'mtx_w' in h_packed_data
        assert 'mtx_w_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_w = h_packed_data['mtx_w']
        mtx_w_score = h_packed_data['mtx_w_score']
        w_emb = self.word_embedding(mtx_w)
        ts_desp = self.e_desp_mtx[mtx_e.view(-1)].view(
            mtx_e.size() + (self.e_desp_mtx.size()[-1],)
        )     # batch, e id, desp word id

        v_desp_words = ts_desp.view(-1)
        ts_desp_emb = self.word_emb(v_desp_words)
        # batch * entity * desp words * word embedding

        ts_desp_emb = ts_desp_emb.view(ts_desp.size() + ts_desp_emb.size()[-1:])

        for cnn, linear in zip(self.l_gloss_cnn, self.l_gloss_linear):
            cnn_emb = self._sentence_cnn(ts_desp_emb, mtx_e, cnn)
            cnn_knrm = self._kernel_vote(cnn_emb, w_emb, mtx_w_score)
            cnn_knrm_score = linear(cnn_knrm).squeeze(-1)
            duet_knrm_score += cnn_knrm_score

        return duet_knrm_score

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


class GlossCNNEmbDuet(DuetKNRM):
    def __init__(self, para, ext_data=None):
        super(GlossCNNEmbDuet, self).__init__(para, ext_data)

        assert ext_data.word_emb is not None
        assert ext_data.entity_desp is not None
        assert para.desp_sent_len
        self.e_desp_mtx = Variable(torch.LongTensor(ext_data.entity_desp[:, :para.desp_sent_len]))
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

    def forward(self, h_packed_data):

        assert 'mtx_e' in h_packed_data
        assert 'mtx_score' in h_packed_data
        assert 'mtx_w' in h_packed_data
        assert 'mtx_w_score' in h_packed_data
        mtx_e = h_packed_data['mtx_e']
        mtx_score = h_packed_data['mtx_score']
        mtx_w = h_packed_data['mtx_w']
        mtx_w_score = h_packed_data['mtx_w_score']
        w_emb = self.word_embedding(mtx_w)
        e_emb = self.embedding(mtx_e)

        # get enriched entity embedding
        ts_desp = self.e_desp_mtx[mtx_e.view(-1)].view(
            mtx_e.size() + (self.e_desp_mtx.size()[-1],)
        )     # batch, e id, desp word id

        v_desp_words = ts_desp.view(-1)
        ts_desp_emb = self.word_emb(v_desp_words)
        # batch * entity * desp words * word embedding

        ts_desp_emb = ts_desp_emb.view(ts_desp.size() + ts_desp_emb.size()[-1:])
        l_e_emb = [e_emb]
        for cnn in self.l_gloss_cnn:
            cnn_emb = self._sentence_cnn(ts_desp_emb, mtx_e, cnn)
            l_e_emb.append(cnn_emb)

        enriched_e_emb = self.emb_merge(
            torch.cat(l_e_emb, dim=-1)
        )

        entity_vote_kernels = self._kernel_scores(enriched_e_emb, mtx_score)
        word_vote_kernels = self._kernel_vote(enriched_e_emb, w_emb, mtx_w_score)

        fuse_vote = torch.cat([entity_vote_kernels, word_vote_kernels], dim=-1)
        output = self.duet_linear(fuse_vote).squeeze(-1)
        return output

    @classmethod
    def _sentence_cnn(cls, ts_desp_emb, mtx_e, cnn):
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
    