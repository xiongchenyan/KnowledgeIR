"""
    embedding cosine with query mean
"""


from knowledge4ir.feature.attention import (
    TermAttentionFeature,
    form_avg_emb,
    calc_query_entity_total_embedding,
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


class EntityEmbeddingAttentionFeature(TermAttentionFeature):
    feature_name_pre = Unicode('EAttEmb')
    l_embedding_in = List(Unicode, default_value=[],
                          help="embedding data inputs, if more than one"
                          ).tag(config=True)
    l_embedding_name = List(Unicode, default_value=[],
                            help="names of corresponding embedding, if more than one"
                            ).tag(config=True)
    tagger = Unicode('tagme', help='tagger').tag(config=True)
    l_features = List(Unicode, default_value=['cosine', 'joint'],
                     help='features: cosine, joint, raw_diff'
                     ).tag(config=True)


    def __init__(self, **kwargs):
        super(EntityEmbeddingAttentionFeature, self).__init__(**kwargs)
        self.l_embedding = []
        self.joint_embedding = None

    def set_external_info(self, external_info):
        super(EntityEmbeddingAttentionFeature, self).set_external_info(external_info)
        self.l_embedding = external_info.l_embedding
        self.l_embedding_name = external_info.l_embedding_name
        self.joint_embedding = external_info.joint_embedding

    def extract(self, h_q_info, l_e):
        """

        :param h_q_info:  query info with everything
        :param l_e: entities
        :return: features for each term: l_h_feature
        """
        l_h_feature = []
        if 'cosine' in self.l_features:
            l_h_this_feature = self._extract_cosine(h_q_info, l_e)
            l_h_feature = mul_update(l_h_feature, l_h_this_feature)

        if 'joint' in self.l_features:
            l_this_h_feature = self._extract_joint( h_q_info, l_e)
            l_h_feature = mul_update(l_h_feature, l_this_h_feature)
        if 'raw_diff' in self.l_features:
            l_this_h_feature = self._extract_raw_diff( h_q_info, l_e)
            l_h_feature = mul_update(l_h_feature, l_this_h_feature)

        return l_h_feature

    def _extract_cosine(self, h_q_info, l_e):
        l_total_h_feature = []
        for name, emb in zip(self.l_embedding_name, self.l_embedding):
            qe_emb = self._calc_e_emb(h_q_info, emb)
            l_this_h_feature = []
            for e in l_e:
                h_feature = {}
                h_feature.update(self._extract_cosine_per_e(h_q_info, e, qe_emb, emb))
                h_feature = dict([(self.feature_name_pre + name + key, score)
                                  for key, score in h_feature.items()])
                l_this_h_feature.append(h_feature)
            l_total_h_feature = mul_update(l_total_h_feature, l_this_h_feature)
        return l_total_h_feature

    def _extract_joint(self, h_q_info, l_e):
        l_h_feature = []
        q_te_join_emb = calc_query_entity_total_embedding(h_q_info, self.joint_embedding)
        for e in l_e:
            h_feature = {}
            h_joint_feature = self._extract_cosine_per_e(h_q_info, e, q_te_join_emb, self.joint_embedding)
            h_feature.update(dict(
                [(item[0] + 'Joint', item[1]) for item in h_joint_feature.items()]
            ))

            h_feature = dict([(self.feature_name_pre + key, score)
                              for key, score in h_feature.items()])
            l_h_feature.append(h_feature)
        return l_h_feature

    def _extract_raw_diff_per_t(self, e, q_emb, emb):
        diff_v = np.random.rand(q_emb.shape[0])
        if (e in emb) & (q_emb is not None):
            diff_v = emb[e] = q_emb
        l_diff = diff_v.tolist()
        h_sim = dict(zip(['diff%03d' % d for d in range(diff_v.shape[0])], l_diff))
        return h_sim

    def _extract_cosine_per_e(self, h_q_info, e, qe_emb, emb):
        h_sim = {}
        if (e not in emb) | (qe_emb is None):
            score = 0
        else:
            score = 1 - cosine(emb[e], qe_emb)
        h_sim['Cos'] = score
        return h_sim

    def _calc_e_emb(self, h_q_info, emb):
        l_e = [ana[0] for ana in h_q_info[self.tagger]['query']]
        qe_emb = form_avg_emb(l_e, emb)
        return qe_emb
