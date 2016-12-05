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

    def set_external_info(self, external_info):
        super(EntityEmbeddingAttentionFeature, self).set_external_info(external_info)
        self.l_embedding = external_info.l_embedding
        self.l_embedding_name = external_info.l_embedding_name

    def extract(self, h_q_info, l_e):
        """

        :param h_q_info:  query info with everything
        :param l_e: entities
        :return: features for each term: l_h_feature
        """
        l_h_feature = []
        for name, emb in zip(self.l_embedding_name, self.l_embedding):
            qe_emb = self._calc_e_emb(h_q_info, emb)
            l_this_h_feature = []
            for e in l_e:
                h_feature = {}
                h_feature.update(self._extract_per_e(h_q_info, e, qe_emb, emb))
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
        l_emb = [emb[e] for e in l_e if e in emb]
        qe_emb = None
        if l_emb:
            qe_emb = np.mean(np.array(l_emb), axis=0)
        return qe_emb
