"""
read pre-trained salience features
in each entity' predict_features field
get feature for each q e
sum up for the final feature
"""

import json
import logging
from knowledge4ir.duet_feature import LeToRFeatureExtractor
from traitlets import (
    Unicode,
    Int,
    List,
    Float,
)
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    sum_pool_feature,
)


class LeToRBOEPreTrainedFeatureExtractor(LeToRFeatureExtractor):
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)
    l_target_fields = List(Unicode,
                           default_value=[TARGET_TEXT_FIELDS],
                           help='doc fields to use'
                           ).tag(config=True)

    feature_name_pre = Unicode('Pretrained')
    default_feature_value = Float(-20, help='filling for empty feature').tag(config=True)
    feature_dim = Int(22,
                      help='number of features in pre-trained')

    def extract(self, qid, docno, h_q_info, h_doc_info):
        l_q_e = [ana['entities'][0]['id'] for ana in h_q_info[self.tagger]['query']]
        for field, l_ana in h_doc_info[self.tagger].items():
            if field not in self.l_target_fields:
                continue
            h_q_e_feature = {}
            for q_e in l_q_e:
                h_q_e_feature[q_e] = [self.default_feature_value] * self.feature_dim
            for ana in l_ana:
                e_id = ana['entities'][0]['id']
                if e_id in h_q_e_feature:
                    l_feature = ana['entities'][0].get('predict_features', [])
                    if l_feature:
                        assert len(l_feature) == self.feature_dim
                        h_q_e_feature[e_id] = l_feature
        l_q_feature = h_q_e_feature.items()
        l_h_q_feature = [dict(zip(
            ['pre_train_%d' % p for p in range(self.feature_dim)],
            q_feature) for q_feature in l_q_feature
        )]
        h_feature = sum_pool_feature(l_h_q_feature, False)

        return h_feature

