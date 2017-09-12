"""
graph translation model
or basic page rank model
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class GraphTranslation(nn.Module):
    """
    input: list of document entities + frequency, target e id
    output: p(target e id is salient)
    """

    def __init__(self, random_walk_step, vocab_size, embedding_dim, pre_embedding=None):
        super(GraphTranslation, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.logistic = nn.Linear(1, 1, bias=True)
        self.random_walk_step = random_walk_step
        if pre_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        return

    def forward(self, v_e, v_score):
        """
        return probability of each one being salient
        :param v_e: the input entity id's
        :param v_score: the initial weights on each entity
        :return: score for each one
        """
        mtx_embedding = self.embedding(v_e)
        mtx_embedding = mtx_embedding.div(
            torch.norm(mtx_embedding, p=2, dim=1).expand_as(mtx_embedding))

        trans_mtx = torch.mm(mtx_embedding, mtx_embedding.transpose(0, 1))
        trans_mtx = trans_mtx.div(
            torch.norm(trans_mtx, p=1, dim=0).expand_as(trans_mtx)
        )
        output = Variable(v_score)
        for p in xrange(self.random_walk_step):
            output = torch.mm(trans_mtx, output)

        output = F.sigmoid(self.logistic(output))
        return output

