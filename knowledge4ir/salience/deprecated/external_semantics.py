import json
import logging

import torch
from torch import nn as nn
from torch.autograd import Variable

from knowledge4ir.salience.external_semantics.description import use_cuda
from knowledge4ir.salience.knrm_vote import KNRM


class DespWordAvgEmbKNRM(KNRM):
    """
    attention weighted word average as entity additional embedding
        word embedding + attention weights -> entity's additional embedding
    """

    def __init__(self, para, ext_data=None):
        super(DespWordAvgEmbKNRM, self).__init__(para, ext_data)
        assert ext_data.word_emb is not None
        assert ext_data.entity_desp is not None

        self.e_att_embedding = nn.Embedding(para.entity_vocab_size,
                                            para.embedding_dim, padding_idx=0)
        if ext_data.entity_emb is not None:
            self.e_att_embedding.weight.data.copy_(torch.from_numpy(ext_data.entity_emb))

        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))

        self.word_att_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                         ext_data.word_emb.shape[1], padding_idx=0)
        self.word_att_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))

        self.emb_merge = nn.Linear(para.embedding_dim + ext_data.word_emb.shape[1], para.embedding_dim, bias=False)

        self.e_desp_mtx = Variable(torch.LongTensor(ext_data.entity_desp))
        logging.info('desp word avg knrm model initialized')
        if use_cuda:
            self.word_att_emb.cuda()
            self.word_emb.cuda()
            self.e_desp_mtx = self.e_desp_mtx.cuda()
            self.e_att_embedding.cuda()
            self.emb_merge.cuda()
            logging.info('model parameters copied to cuda')

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
        ts_desp = self.e_desp_mtx[mtx_e.data.view(-1)].view(
            mtx_e.size() + (self.e_desp_mtx.size()[-1],)
        )     # batch, e id, desp word id
        v_desp_words = ts_desp.view(-1)
        ts_desp_content_emb = self.word_emb(v_desp_words)
        ts_desp_att_emb = self.word_att_emb(v_desp_words)    # words * emb
        # reshape to batch, e id, desp word id, emb
        ts_desp_content_emb = ts_desp_content_emb.view(ts_desp.size() + ts_desp_content_emb.size()[-1:])
        ts_desp_att_emb = ts_desp_att_emb.view(ts_desp.size() + ts_desp_att_emb.size()[-1:])
        return ts_desp_att_emb, ts_desp_content_emb

    def _att_avg_emb(self, mtx_att_embedding, ts_desp_att_emb, ts_desp_content_emb):
        """

        :param mtx_att_embedding: batch * entity * emb
        :param ts_desp_att_emb:   batch * entity * desp words * emb
        :param ts_desp_content_emb: batch * entity * desp words * emb
        :return:
        """
        logging.debug('mtx_att_embedding shape %s', json.dumps(mtx_att_embedding.size()))
        logging.debug('ts_desp_att_emb shape %s', json.dumps(ts_desp_att_emb.size()))
        logging.debug('ts_desp_content_emb shape %s', json.dumps(ts_desp_content_emb.size()))

        # batch, e id, desp word' weights
        att_score = torch.matmul(
            ts_desp_att_emb, mtx_att_embedding.unsqueeze(-1)).squeeze(-1)
        logging.debug('att_score shape %s', json.dumps(att_score.size()))
        att_score = nn.functional.softmax(att_score)
        att_word_emb = torch.matmul(
            att_score.unsqueeze(-2), ts_desp_content_emb).squeeze(-2)   # avg desp word emb for each entity
        logging.debug('att_word_emb shape %s', json.dumps(att_word_emb.size()))
        return att_word_emb


class DespSentRNNEmbedKNRM(KNRM):
    """
    rnn of the description's first 10 words
    """

    def __init__(self, para, ext_data=None):
        super(DespSentRNNEmbedKNRM, self).__init__(para, ext_data)

        assert ext_data.word_emb is not None
        assert ext_data.entity_desp is not None
        assert para.desp_sent_len
        self.e_desp_mtx = Variable(torch.LongTensor(ext_data.entity_desp[:, :para.desp_sent_len]))
        # self.e_desp_mtx = self.e_desp_mtx[:, :para.desp_sent_len]
        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))

        self.desp_rnn = torch.nn.GRU(
            input_size=para.embedding_dim,
            hidden_size=para.embedding_dim,
            num_layers=1,
            bias=False,
            batch_first=True,
            bidirectional=True
        )
        self.emb_merge = nn.Linear(
            para.embedding_dim * 3,
            para.embedding_dim,
            bias=False
        )
        if use_cuda:
            self.desp_rnn.cuda()
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

        # reshape for RNN:
        # now is (batch * entity) * desp's words * word embedding
        ts_desp_emb = ts_desp_emb.view((-1,) + ts_desp_emb.size()[-2:])

        h0 = Variable(torch.randn(2, ts_desp_emb.size()[0], ts_desp_emb.size()[-1]))
        if use_cuda:
            h0 = h0.cuda()
        logging.debug('starting the bi-gru with shape %s', json.dumps(h0.size()))
        logging.debug('and input sequence shape %s', json.dumps(ts_desp_emb.size()))
        __, desp_rnn_out = self.desp_rnn(ts_desp_emb, h0)
        logging.debug('rnn out shape %s', json.dumps(desp_rnn_out.size()))
        # desp_rnn_out = desp_rnn_out.transpose(0, 1)
        forward_rnn_out = desp_rnn_out[0, :, :].contiguous()  # now batch-embedding
        backward_rnn_out = desp_rnn_out[1, :, :].contiguous()
        logging.debug('one direction rnn out shape %s', json.dumps(forward_rnn_out.size()))
        forward_rnn_out = forward_rnn_out.view(mtx_e.size() + forward_rnn_out.size()[-1:])
        backward_rnn_out = backward_rnn_out.view_as(forward_rnn_out)
        enriched_e_embedding = self.emb_merge(
            torch.cat((mtx_embedding, forward_rnn_out, backward_rnn_out), dim=-1)
        )

        return self._knrm_opt(enriched_e_embedding, mtx_score)


class LinearGlossCNNEmbedKNRM(KNRM):

    def __init__(self, para, ext_data=None):
        super(LinearGlossCNNEmbedKNRM, self).__init__(para, ext_data)

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

        # now is (batch * entity) * desp's words * word embedding

        knrm_score = self._kernel_scores(mtx_embedding, mtx_score)
        knrm_output = self.linear(knrm_score).squeeze(-1)

        for cnn_emb in l_cnn_emb:
            cnn_knrm_score = self._kernel_scores(cnn_emb, mtx_score).squeeze(-1)
            knrm_output += cnn_knrm_score

        return knrm_output

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


class GlossCNNEmbedKNRM(KNRM):
    """
    Cnn of the description's first 20 words
    """

    def __init__(self, para, ext_data=None):
        super(GlossCNNEmbedKNRM, self).__init__(para, ext_data)

        assert ext_data.word_emb is not None
        assert ext_data.entity_desp is not None
        assert para.desp_sent_len
        self.e_desp_mtx = Variable(torch.LongTensor(ext_data.entity_desp[:, :para.desp_sent_len]))
        # self.e_desp_mtx = self.e_desp_mtx[:, :para.desp_sent_len]
        self.word_emb = nn.Embedding(ext_data.word_emb.shape[0],
                                     ext_data.word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(ext_data.word_emb))

        self.gloss_cnn = torch.nn.Conv1d(
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
            self.gloss_cnn.cuda()
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

        # reshape for CNN:
        # now is (batch * entity) * desp's words * word embedding
        ts_desp_emb = ts_desp_emb.view((-1,) + ts_desp_emb.size()[-2:])
        ts_desp_emb = ts_desp_emb.transpose(-1, -2)   # now batch * embedding * words
        logging.debug('cnn input sequence shape %s', json.dumps(ts_desp_emb.size()))
        cnn_filter = self.gloss_cnn(ts_desp_emb)
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
        enriched_e_embedding = self.emb_merge(
            torch.cat((mtx_embedding, cnn_emb), dim=-1)
        )

        return self._knrm_opt(enriched_e_embedding, mtx_score)