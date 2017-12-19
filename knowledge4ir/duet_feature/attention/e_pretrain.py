"""
load pretrained features as query entity attention features
"""

import logging
from traitlets import (
    List,
    Unicode,
    Int,
    Float,
)

from knowledge4ir.duet_feature.attention import (
    EntityAttentionFeature,
    mul_update,
)


class EntityPretrainAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode('Pretrain')
    prf_d = Int(20).tag(config=True)
    tagger = Unicode('spot', help="tagger").tag(config=True)
    pretrain_feature_field = Unicode('salience_feature', help='field of trained features').tag(config=True)
    feature_dim = Int(22,
                      help='number of features in pre-trained').tag(config=True)
    default_feature_value = Float(-20, help='filling for empty feature').tag(config=True)
    # mode = Unicode('full', help='full|lean').tag(config=True)

    def __init__(self, **kwargs):
        super(EntityPretrainAttentionFeature, self).__init__(**kwargs)
        self.h_q_rank_info = {}
        self.h_surface_info = {}

    def extract(self, h_q_info, l_e):
        l_h_feature = []

        l_h_feature = mul_update(l_h_feature, self._extract_raw_pretrain(h_q_info, l_e))
        return l_h_feature

    def _extract_raw_pretrain(self, h_q_info, l_e):
        l_h_feature = []
        h_e_feature = {}
        l_ana = h_q_info.get(self.tagger, {}).get('query', [])
        l_default_feature = [self.default_feature_value] * self.feature_dim
        l_f_name = ['%s_%s_%03d' % (self.feature_name_pre, self.pretrain_feature_field, p)
                    for p in xrange(self.feature_dim)]
        for ana in l_ana:
            e_id = ana['entities'][0]['id']
            l_feature = ana['entities'][0].get(self.pretrain_feature_field, [])
            if l_feature:
                assert len(l_feature) == self.feature_dim
                h_e_feature[e_id] = dict(zip(l_f_name, l_feature))
        for e in l_e:
            if e in h_e_feature:
                l_h_feature.append(h_e_feature[e])
            else:
                l_h_feature.append(
                    dict(zip(l_f_name, list(l_default_feature)))
                )
        return l_h_feature

