"""
    embedding cosine with query mean
"""

from knowledge4ir.feature.attention import (
    TermAttentionFeature,
    calc_query_entity_total_embedding,
    form_avg_emb,
    mul_update,
)
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
    l_features = List(Unicode, default_value=['cosine', 'joint'],
                      help='features: cosine, joint, raw_diff'
                      ).tag(config=True)
    
    def __init__(self, **kwargs):
        super(TermEmbeddingAttentionFeature, self).__init__(**kwargs)
        self.word2vec = None
        self.joint_embedding = None

    def set_external_info(self, external_info):
        super(TermEmbeddingAttentionFeature, self).set_external_info(external_info)
        self.word2vec = external_info.word2vec
        self.joint_embedding = external_info.joint_embedding

    def extract(self, h_q_info, l_t):
        """

        :param h_q_info:  query info with everything
        :param l_t: terms
        :return: features for each term: l_h_feature
        """
        # assert self.word2vec

        l_h_feature = []
        if 'cosine' in self.l_features:
            l_this_h_feature = self._extract_cosine( h_q_info, l_t)
            l_h_feature = mul_update(l_h_feature, l_this_h_feature)
        if 'joint' in self.l_features:
            l_this_h_feature = self._extract_joint( h_q_info, l_t)
            l_h_feature = mul_update(l_h_feature, l_this_h_feature)
        if 'raw_diff' in self.l_features:
            l_this_h_feature = self._extract_raw_diff( h_q_info, l_t)
            l_h_feature = mul_update(l_h_feature, l_this_h_feature)

        return l_h_feature

    def _extract_cosine(self, h_q_info, l_t):
        emb = self.word2vec
        q_emb = self._calc_q_emb(h_q_info, emb)
        l_this_h_feature = []
        for t in l_t:
            h_feature = {}
            h_feature.update(self._extract_cosine_per_t(t, q_emb, emb))
            h_feature = dict([(self.feature_name_pre + key, score)
                              for key, score in h_feature.items()])
            l_this_h_feature.append(h_feature)
        return l_this_h_feature

    def _extract_joint(self,  h_q_info, l_t):
        q_te_joint_emb = calc_query_entity_total_embedding(h_q_info, self.joint_embedding)
        l_this_h_feature = []
        for t in l_t:
            h_feature = {}
            h_joint_feature = self._extract_cosine_per_t(t, q_te_joint_emb, self.joint_embedding)
            h_feature.update(dict(
                [(item[0] + 'Joint', item[1]) for item in h_joint_feature.items()]
            ))

            h_feature = dict([(self.feature_name_pre + key, score) for key, score in h_feature.items()])
            l_this_h_feature.append(h_feature)
        return l_this_h_feature

    def _extract_raw_diff(self, h_q_info, l_t):
        emb = self.word2vec
        q_emb = self._calc_q_emb(h_q_info, emb)
        l_this_h_feature = []
        for t in l_t:
            h_feature = {}
            h_feature.update(self._extract_raw_diff_per_t(t, q_emb, emb))
            h_feature = dict([(self.feature_name_pre + key, score)
                              for key, score in h_feature.items()])
            l_this_h_feature.append(h_feature)
        return l_this_h_feature

    def _extract_cosine_per_t(self, t, q_emb, emb):
        h_sim = {}
        if (t not in emb) | (q_emb is None):
            score = 0
        else:
            score = 1 - cosine(emb[t], q_emb)
        h_sim['Cos'] = score
        return h_sim

    def _extract_raw_diff_per_t(self, t, q_emb, emb):
        diff_v = np.random.rand(q_emb.shape[0])
        if (t in emb) & (q_emb is not None):
            diff_v = emb[t] = q_emb
        l_diff = diff_v.tolist()
        h_sim = dict(zip(['diff%03d' % d for d in range(diff_v.shape[0])], l_diff))
        return h_sim

    def _calc_q_emb(self, h_q_info, emb):
        l_q_t = h_q_info['query'].lower().split()
        q_emb = form_avg_emb(l_q_t, emb)
        return q_emb


    


