"""
read pre-trained salience features
in each entity' predict_features field
get feature for each q e
sum up for the final feature
"""

import json
import logging
import numpy as np
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
                           default_value=TARGET_TEXT_FIELDS,
                           help='doc fields to use'
                           ).tag(config=True)

    feature_name_pre = Unicode('Pretrained')
    default_feature_value = Float(-20, help='filling for empty feature').tag(config=True)
    feature_dim = Int(22,
                      help='number of features in pre-trained').tag(config=True)
    pretrain_feature_field = Unicode('salience_feature', help='field of trained features').tag(config=True)
    normalize_feature = Unicode(
        help='whether and how to normalize feature. Currently supports softmax, minmax, uniq, doclen, expuniq, docuniq'
    ).tag(config=True)

    def extract(self, qid, docno, h_q_info, h_doc_info):
        l_q_e = [ana['entities'][0]['id'] for ana in h_q_info[self.tagger]['query']]
        h_feature = dict()
        for field, l_ana in h_doc_info[self.tagger].items():
            if field not in self.l_target_fields:
                continue
            h_q_e_feature = {}
            for q_e in l_q_e:
                h_q_e_feature[q_e] = [self.default_feature_value] * self.feature_dim
            h_e_feature = {}
            for ana in l_ana: # get features for all entities
                e_id = ana['entities'][0]['id']
                l_feature = ana['entities'][0].get(self.pretrain_feature_field, [])
                if l_feature:
                    assert len(l_feature) == self.feature_dim
                    h_e_feature[e_id] = l_feature
            if self.normalize_feature:   # normalize feature
                l_e_ll_feature = h_e_feature.items()
                ll_feature = [item[1] for item in l_e_ll_feature]
                l_e = [item[0] for item in l_e_ll_feature]
                ll_feature = self._normalize_feature(ll_feature)
                h_e_feature = dict(zip(l_e, ll_feature))
            for q_e in l_q_e:
                if q_e in h_e_feature:
                    h_q_e_feature[q_e] = h_e_feature[q_e]
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

    def _normalize_feature(self, ll_feature):
        """
        normalize feature
        :param ll_feature:
        :return:
        """
        if not ll_feature:
            return ll_feature

        h_norm = {
            "softmax": self._softmax_feature,
            'minmax': self._minmax_feature,
            'uniq': self._uniq_e_normalize_feature,
            'doclen': self._doc_len_normalize_feature,
            'expuniq': self._exp_uniq_e_normalize_feature,
            'docuniq': self._doc_uniq_normalize_feature,
        }
        if self.normalize_feature not in h_norm:
            logging.info('normalize via [%s] not implemented', self.normalize_feature)
            raise NotImplementedError
        return h_norm[self.normalize_feature](ll_feature)

    def _softmax_feature(self, ll_feature):
        m_feature = np.array(ll_feature)
        exp_feature = np.exp(m_feature)
        sum_norm = np.sum(exp_feature, axis=0)
        normalized_e = exp_feature / sum_norm
        ll_normalized_feature = np.log(normalized_e).tolist()
        return ll_normalized_feature

    def _minmax_feature(self, ll_feature):
        m_feature = np.array(ll_feature)
        max_feature = np.amax(m_feature, axis=0)
        min_feature = np.amin(m_feature, axis=0)
        z_feature = np.maximum(max_feature - min_feature, 1e-10)
        normalized_feature = (m_feature - min_feature) / z_feature
        return normalized_feature.tolist()

    def _uniq_e_normalize_feature(self, ll_feature):
        m_feature = np.array(ll_feature)
        m_feature /= float(m_feature.shape[0])
        return m_feature.tolist()

    def _exp_uniq_e_normalize_feature(self, ll_feature):
        m_feature = np.array(ll_feature)
        z = float(m_feature.shape[0])
        m_feature = np.log(np.exp(m_feature) / float(z))
        return m_feature.tolist()

    def _doc_len_normalize_feature(self, ll_feature):
        m_feature = np.array(ll_feature)
        z = np.sum(np.exp(m_feature[:, 0]))
        m_feature = np.log(np.exp(m_feature) / float(z))
        return m_feature.tolist()

    def _doc_uniq_normalize_feature(self, ll_feature):
        m_feature = np.array(ll_feature)
        z = np.sum(np.exp(m_feature[:, 0]))
        m_feature = np.log(np.exp(m_feature) / float(z) / float(m_feature.shape[0]))
        return m_feature.tolist()