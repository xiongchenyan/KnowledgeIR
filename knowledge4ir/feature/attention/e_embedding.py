"""
    embedding cosine with query mean
"""


from knowledge4ir.feature.attention import (
    TermAttentionFeature,
    form_avg_emb,
    calc_query_entity_total_embedding,
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
        for name, emb in zip(self.l_embedding_name, self.l_embedding):
            # qe_emb = self._calc_e_emb(h_q_info, emb)
            q_te_join_emb = calc_query_entity_total_embedding(h_q_info, self.joint_embedding)
            l_this_h_feature = []
            for e in l_e:
                h_feature = {}
                # h_feature.update(self._extract_per_e(h_q_info, e, qe_emb, emb))
                h_joint_feature = self._extract_per_e(h_q_info, e, q_te_join_emb, emb)
                h_feature.update(dict(
                    [(item[0] + 'Joint', item[1]) for item in h_joint_feature.items()]
                ))

                h_feature = dict([(self.feature_name_pre + name + key, score)
                                  for key, score in h_feature.items()])
                l_this_h_feature.append(h_feature)
            if not l_h_feature:
                l_h_feature = l_this_h_feature
            else:
                for p in xrange(len(l_h_feature)):
                    l_h_feature[p].update(l_this_h_feature[p])

        return l_h_feature

    def _extract_per_e(self, h_q_info, e, qe_emb, emb):
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
