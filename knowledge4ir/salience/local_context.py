"""
local context models
used data:
    mtx_e: batched d-e id's
    ts_local_context: l_sentence word id mtx for each entity
    pre_trained word embedding
    entity embedding
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from knowledge4ir.salience.utils import SalienceBaseModel
import logging
import json
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()


class LocalAvgWordVotes(SalienceBaseModel):

    def __init__(self, para, pre_emb=None):
        super(LocalAvgWordVotes, self).__init__(para, pre_emb)
        assert para.word_emb_in
        self.embedding = nn.Embedding(para.entity_vocab_size,
                                      para.embedding_dim, padding_idx=0)

        if pre_emb:
            logging.info('copying entity embedding to model...')
            self.embedding.weight.data.copy_(torch.from_numpy(pre_emb))

        logging.info('loading pre trained word emb from [%s]...', para.word_emb_in)
        word_emb = np.load(open(para.word_emb_in))
        self.word_embedding = nn.Embedding(word_emb.shape[0],
                                           word_emb.shape[1],
                                           padding_idx=0,
                                           )
        assert word_emb.shape[1] == para.embedding_dim
        self.word_embedding.weight.data.copy_(torch.from_numpy(word_emb))
        self.word_embedding.requires_grad = False   # not training the word embedding
        self.linear_combine = nn.Linear(2, 1)   # combine the max pool and sum pool of votes

        if use_cuda:
            logging.info('transferring model to gpu...')
            self.embedding.cuda()
            self.linear_combine.cuda()
            self.word_embedding.cuda()

    def forward(self, h_packed_data):
        assert 'mtx_e' in h_packed_data
        assert 'ts_local_context' in h_packed_data

        mtx_e = h_packed_data['mtx_e']
        ts_context = h_packed_data['ts_local_context']

        ts_e_embedding = self.embedding(mtx_e)   # batch-doc-e-embedding
        ts_e_sent_word_embedding = self.word_embedding(ts_context)   # batch-doc-e-sent-word-word2vec
        logging.debug('sent emb size: %s', json.dumps(ts_e_sent_word_embedding.size()))
        # batch-doc-e-sent-embedding
        ts_e_sent_embedding = torch.mean(ts_e_sent_word_embedding, dim=-2, keepdim=False)

        ts_e_embedding = ts_e_embedding.unsqueeze(-1)
        ts_e_sent_vote = torch.matmul(ts_e_sent_embedding, ts_e_embedding).squeeze(-1)
        #  ts_e_sent_vote is a batch-doc-e-sent tensor, last dim is the vote from each local sent
        logging.debug('sent voting size: %s', json.dumps(ts_e_sent_vote.size()))

        ts_e_sum = torch.sum(ts_e_sent_vote, dim=-1, keepdim=True)
        ts_e_max = torch.max(ts_e_sent_vote, dim=-1, keepdim=True)

        ts_e_voteFeature = torch.cat(ts_e_sum, ts_e_max, -1)
        logging.debug('ts_e_voteFeature size: %s', json.dumps(ts_e_voteFeature.size()))
        # batch-doc-e-[feature: max and mean]
        output = F.tanh(self.linear_combine(ts_e_voteFeature).squeeze(-1))
        logging.debug('output size: %s', json.dumps(ts_e_voteFeature.size()))
        return output

    def save_model(self, output_name):
        logging.info('saving embedding and linear combine weights to [%s]', output_name)
        emb_mtx = self.embedding.weight.data.cpu().numpy()
        np.save(open(output_name + '.emb.npy', 'w'), emb_mtx)
        np.save(open(output_name + '.linear_combine.npy', 'w'),
                self.linear_combine.weight.data.cpu().numpy())




