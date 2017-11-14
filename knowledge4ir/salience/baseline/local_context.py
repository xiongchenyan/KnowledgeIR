"""
local context models
used data:
    mtx_e: batched d-e id's
    ts_local_context: l_sentence word id mtx for each entity
    pre_trained word embedding
    entity embedding
does not work 11/14/2017
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from knowledge4ir.salience.base import SalienceBaseModel
import logging
import json
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()


class LocalAvgWordVotes(SalienceBaseModel):
    final_combine_dim = 2   # sum and mean of local votes

    def __init__(self, para, pre_emb=None):
        super(LocalAvgWordVotes, self).__init__(para, pre_emb)
        assert para.word_emb_in
        self.embedding = nn.Embedding(para.entity_vocab_size,
                                      para.embedding_dim, padding_idx=0)
        self.embedding_dim = para.embedding_dim
        if pre_emb is not None:
            logging.info('copying entity embedding to model...')
            self.embedding.weight.data.copy_(torch.from_numpy(pre_emb))

        logging.info('loading pre trained word emb from [%s]...', para.word_emb_in)
        word_emb = np.load(open(para.word_emb_in))
        logging.info('loaded word emb with shape %s', json.dumps(word_emb.shape))
        self.word_embedding = nn.Embedding(word_emb.shape[0],
                                           word_emb.shape[1],
                                           padding_idx=0,
                                           )
        assert word_emb.shape[1] == para.embedding_dim
        self.word_embedding.weight.data.copy_(torch.from_numpy(word_emb))
        self.word_embedding.requires_grad = para.train_word_emb   # not training the word embedding
        self.linear_combine = nn.Linear(self.final_combine_dim, 1)   # combine the max pool and sum pool of votes

        if use_cuda:
            logging.info('transferring model to gpu...')
            self.embedding.cuda()
            self.linear_combine.cuda()
            self.word_embedding.cuda()

    def forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        ts_context = h_packed_data['ts_local_context']  # batch-doc-e-sent-word

        ts_e_embedding = self.embedding(mtx_e)   # batch-doc-e-embedding

        ts_e_sent_word_embedding = self.word_embedding(
            ts_context.view(-1, ts_context.size()[-1])
        ).view(ts_context.size() + (self.embedding_dim, ))
        # batch-doc-e-sent-word-word2vec
        # batch-doc-e-sent-embedding
        ts_e_sent_embedding = torch.mean(ts_e_sent_word_embedding, dim=-2, keepdim=False)

        ts_e_sum, ts_e_max = self.sent_vote(ts_e_embedding, ts_e_sent_embedding)
        ts_e_voteFeature = torch.cat((ts_e_sum, ts_e_max), -1)

        # batch-doc-e-[feature: max and mean]
        output = F.tanh(self.linear_combine(ts_e_voteFeature).squeeze(-1))
        return output

    def save_model(self, output_name):
        logging.info('saving embedding and linear combine weights to [%s]', output_name)
        emb_mtx = self.embedding.weight.data.cpu().numpy()
        np.save(open(output_name + '.emb.npy', 'w'), emb_mtx)
        word_emb_mtx = self.word_embedding.weight.data.cpu().numpy()
        np.save(open(output_name + '.word_emb.npy', 'w'), word_emb_mtx)
        np.save(open(output_name + '.linear_combine.npy', 'w'),
                self.linear_combine.weight.data.cpu().numpy())

    def _assert_input(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'ts_local_context' in h_packed_data

    def sent_vote(self, ts_e_embedding, ts_e_sent_embedding):
        ts_e_embedding = ts_e_embedding.unsqueeze(-1)
        ts_e_sent_vote = torch.matmul(ts_e_sent_embedding, ts_e_embedding).squeeze(-1)
        #  ts_e_sent_vote is a batch-doc-e-sent tensor, last dim is the vote from each local sent

        # logging.debug('sent voting size: %s', json.dumps(ts_e_sent_vote.size()))

        ts_e_sum = torch.sum(ts_e_sent_vote, dim=-1, keepdim=True)
        ts_e_max = torch.max(ts_e_sent_vote, dim=-1, keepdim=True)[0]
        return ts_e_sum, ts_e_max


class LocalRNNVotes(LocalAvgWordVotes):
    final_combine_dim = 4   # sum, max of forward and backward GRU votes

    def __init__(self, para, pre_emb=None):
        super(LocalRNNVotes, self).__init__(para, pre_emb)
        self.rnn = torch.nn.GRU(
            input_size=para.embedding_dim,
            hidden_size=para.embedding_dim,
            num_layers=1,
            bias=False,
            batch_first=True,
            bidirectional=True
        )
        if use_cuda:
            self.rnn.cuda()

    def forward(self, h_packed_data):
        self._assert_input(h_packed_data)
        mtx_e = h_packed_data['mtx_e']
        ts_context = h_packed_data['ts_local_context']

        forward_rnn_out, backward_rnn_out = self._get_rnns(ts_context)

        ts_e_embedding = self.embedding(mtx_e)
        forward_sent_vote_sum, forward_sent_vote_mean = self.sent_vote(
            ts_e_embedding, forward_rnn_out
        )
        backward_sent_vote_sum, backward_sent_vote_mean = self.sent_vote(
            ts_e_embedding, backward_rnn_out
        )

        ts_e_voteFeature = torch.cat((
            forward_sent_vote_sum, forward_sent_vote_mean,
            backward_sent_vote_sum, backward_sent_vote_mean), -1)

        output = F.tanh(self.linear_combine(ts_e_voteFeature)).squeeze(-1)
        return output

    def _get_rnns(self, ts_context):
        ts_e_sent_word_embedding = self.word_embedding(
            ts_context.view(-1, ts_context.size()[-1])
        ).view(ts_context.size() + (self.embedding_dim, ))
        # batch-doc-e-sent-word-word2vec

        batch_rnn_input = ts_e_sent_word_embedding.view(
            (-1,) + ts_e_sent_word_embedding.size()[-2:]
        )
        h0 = Variable(torch.randn(2, batch_rnn_input.size()[0], self.embedding_dim))
        if use_cuda:
            h0 = h0.cuda()
        __, rnn_out = self.rnn(batch_rnn_input, h0)
        rnn_out = rnn_out.transpose(0, 1)
        forward_rnn_out = rnn_out[:, 0, :].contiguous()  # now batch-doc-e-sent-embedding
        backward_rnn_out = rnn_out[:, 1, :].contiguous()

        # logging.debug('rnn out shape %s', json.dumps(forward_rnn_out.size()))

        forward_rnn_out = forward_rnn_out.view(
            ts_e_sent_word_embedding.size()[:-2] + forward_rnn_out.size()[-1:])
        backward_rnn_out = backward_rnn_out.view(
            ts_e_sent_word_embedding.size()[:-2] + backward_rnn_out.size()[-1:])
        # logging.debug('reshaped to %s', json.dumps(forward_rnn_out.size()))
        return forward_rnn_out, backward_rnn_out

    def save_model(self, output_name):
        super(LocalRNNVotes, self).save_model(output_name)
        for name, weights in self.rnn._parameters.items():
            mtx = weights.data.cpu().numpy()
            np.save(open(output_name + '.' + name + '.npy', 'w'),
                    mtx)


class LocalRNNMaxSim(LocalRNNVotes):
    final_combine_dim = 2

    def forward(self, h_packed_data):
        self._assert_input(h_packed_data)
        mtx_e = h_packed_data['mtx_e']
        ts_context = h_packed_data['ts_local_context']

        forward_rnn_out, backward_rnn_out = self._get_rnns(ts_context)

        max_forward = torch.max(forward_rnn_out, dim=-2, keepdim=False)[0]
        max_backward = torch.max(backward_rnn_out, dim=-2, keepdim=False)[0]
        # logging.debug('maxpooled forward rnn shape: %s',
        #               json.dumps(max_forward.size()))
        ts_e_embedding = self.embedding(mtx_e)

        forward_sim = torch.sum(
            max_forward * ts_e_embedding,
            dim=-1,
            keepdim=True
        )
        backward_sim = torch.sum(
            max_backward * ts_e_embedding,
            dim=-1,
            keepdim=True
        )

        ts_e_voteFeature = torch.cat(
            (forward_sim, backward_sim,),
            -1
        )
        output = F.tanh(self.linear_combine(ts_e_voteFeature)).squeeze(-1)
        return output



