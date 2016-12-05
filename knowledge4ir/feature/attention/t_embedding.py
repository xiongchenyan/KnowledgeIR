"""
    embedding cosine with query mean
"""

from knowledge4ir.feature.attention import TermAttentionFeature
import json
import logging
from traitlets import (
    List,
    Unicode,
    Int
)
import numpy as np
from scipy.spatial.distance import cosine


class TermEmbeddingAttentionFeature(TermAttentionFeature):
    feature_name_pre = Unicode('TAttEmb')
    
    def __init__(self, **kwargs):
        super(TermEmbeddingAttentionFeature, self).__init__(**kwargs)
        self.word2vec = None

    def set_external_info(self, external_info):
        super(TermEmbeddingAttentionFeature, self).set_external_info(external_info)
        self.word2vec = external_info.word2vec

    def extract(self, h_q_info, l_t):
        """

        :param h_q_info:  query info with everything
        :param l_t: terms
        :return: features for each term: l_h_feature
        """
        assert self.word2vec

        l_h_feature = []
        emb = self.word2vec
        q_emb = self._calc_q_emb(h_q_info, emb)
        l_this_h_feature = []
        for t in l_t:
            h_feature = {}
            h_feature.update(self._extract_per_t(h_q_info, t, q_emb, emb))
            h_feature = dict([(self.feature_name_pre + key, score) for key, score in h_feature.items()])
            l_this_h_feature.append(h_feature)
        if not l_h_feature:
            l_h_feature = l_this_h_feature
        else:
            for p in xrange(len(l_h_feature)):
                l_h_feature[p].update(l_this_h_feature[p])

        return l_h_feature

    def _extract_per_t(self, h_q_info, t, q_emb, emb):
        h_sim = {}
        if t not in emb:
            score = 0
        else:
            score = 1 - cosine(emb[t], q_emb)
        h_sim['Cos'] = score
        return h_sim

    def _calc_q_emb(self, h_q_info, emb):
        l_q_t = h_q_info['query'].split()
        l_emb = [emb[t] for t in l_q_t if t in emb]
        q_emb = np.mean(np.array(l_emb), axis=0)
        return q_emb



    


