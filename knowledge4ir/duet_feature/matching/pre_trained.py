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
    body_field,
)


class LeToRBOEPreTrainedFeatureExtractor(LeToRFeatureExtractor):
    tagger = Unicode('spot', help='tagger used, as in q info and d info'
                     ).tag(config=True)
    l_target_fields = List(Unicode,
                           default_value=[body_field],
                           help='doc fields to use'
                           ).tag(config=True)

    feature_name_pre = Unicode('Pretrained')
    default_feature_value = Float(-20, help='filling for empty feature').tag(config=True)
    feature_dim = Int(22,
                      help='number of features in pre-trained').tag(config=True)
    pretrain_feature_field = Unicode('salience_feature', help='field of trained features').tag(config=True)

    def extract(self, qid, docno, h_q_info, h_doc_info):
        l_q_e = [ana['entities'][0]['id'] for ana in h_q_info[self.tagger]['query']]
        h_feature = dict()
        for field, l_ana in h_doc_info[self.tagger].items():
            if field not in self.l_target_fields:
                continue
            h_q_e_feature = {}
            for q_e in l_q_e:
                h_q_e_feature[q_e] = [self.default_feature_value] * self.feature_dim
            for ana in l_ana:
                e_id = ana['entities'][0]['id']
                if e_id in h_q_e_feature:
                    l_feature = ana['entities'][0].get(self.pretrain_feature_field, [])
                    if l_feature:
                        assert len(l_feature) == self.feature_dim
                        h_q_e_feature[e_id] = l_feature
            l_q_feature = [item[1] for item in h_q_e_feature.items()]
            l_h_q_feature = []
            for l_feature in l_q_feature:
                l_name = ['%s_%s_%03d' % (field, self.pretrain_feature_field, p)
                          for p in range(self.feature_dim)]
                h_this_f = dict(zip(l_name, l_feature))
                # logging.info('name %s', json.dumps(l_name))
                # logging.info('feature %s', json.dumps(l_feature))
                l_h_q_feature.append(h_this_f)

            # l_h_q_feature = [dict(zip(
            #     ['%s_pre_train_%d' % (field, p) for p in range(self.feature_dim)],
            #     q_feature) for q_feature in l_q_feature
            # )]
            h_feature.update(sum_pool_feature(l_h_q_feature, False))

        return h_feature

