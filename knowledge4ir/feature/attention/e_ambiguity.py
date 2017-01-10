"""
model the ambiguity of the surface form->e

input:
    external_info: surface form top 5 dict
    prf stuff
output:
    is top candidate
    difference with second 2, if not top, then is 0
    entropy of top 5

    prf: # of other possible linked e
    prf: prob of other possible linked e

"""


from knowledge4ir.feature.attention import (
    EntityAttentionFeature,
    mul_update,
)
from scipy.stats import entropy
from knowledge4ir.feature.attention.t_prf import TermPrfAttentionFeature
import json
import logging
import math
from traitlets import (
    List,
    Unicode,
    Int
)
from knowledge4ir.utils import (
    body_field,
    TARGET_TEXT_FIELDS,
    term2lm,
    # rm3,
)


class EntityAmbiguityAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode('Ambi')
    prf_d = Int(20).tag(config=True)
    tagger = Unicode('tagme', help="tagger").tag(config=True)
    l_feature = List(Unicode, default_value=['surface', 'prf']).tag(config=True)

    def __init__(self, **kwargs):
        super(EntityAmbiguityAttentionFeature, self).__init__(**kwargs)
        # self.h_field_h_df = {}
        # self.h_corpus_stat = {}
        self.h_q_rank_info = {}
        self.h_surface_info = {}

    def set_external_info(self, external_info):
        super(EntityAmbiguityAttentionFeature, self).set_external_info(external_info)
        self.h_q_rank_info = dict(external_info.ll_q_rank_info)
        self.h_surface_info = external_info.h_surface_info

    def extract(self, h_q_info, l_e):
        l_h_feature = []
        if 'surface' in self.l_feature:
            l_h_feature = mul_update(l_h_feature, self._extract_sf_feature(h_q_info, l_e))
        if 'prf' in self.l_feature:
            l_h_feature = mul_update(l_h_feature, self._extract_prf_feature(h_q_info, l_e))
        return l_h_feature

    def _extract_sf_feature(self, h_q_info, l_e):
        """

        :param h_q_info:
        :param l_e:
        :return:
        """
        l_h_feature = []
        for e in l_e:
            l_h_feature.append(self._extract_per_e_sf(h_q_info, e))
        return l_h_feature

    def _extract_prf_feature(self, h_q_info, l_e):
        l_h_feature = []
        for e in l_e:
            l_h_feature.append(self._extract_per_e_prf(h_q_info, e))
        return l_h_feature

    def _extract_per_e_sf(self, h_q_info, e):
        h_feature = dict()
        p = self._find_entity_p(h_q_info, e)
        ana = h_q_info[self.tagger]['query'][p]

        sf = h_q_info['query'][ana[1]:ana[2]]

        if sf not in self.h_surface_info:
            logging.warn('surface [%s] not found in dict', sf)
            return {}

        l_top_k = self.h_surface_info[sf]
        l_candidate_prob = l_top_k
        z = float(sum([item[1] for item in l_candidate_prob]))
        l_candidate_prob = [(item[0], item[1] / z) for item in l_candidate_prob]
        l_candidate_prob.sort(key=lambda item: -item[1])
        is_top = 0
        if l_candidate_prob[0][0] == e:
            is_top = 1
        h_feature[self.feature_name_pre + 'IsTop'] = is_top

        margin = 0
        if is_top:
            if len(l_candidate_prob) < 2:
                margin = 1
            else:
                margin = l_candidate_prob[0][1] - l_candidate_prob[1][1]
        h_feature[self.feature_name_pre + 'Margin'] = margin

        link_entropy = entropy([item[1] for item in l_candidate_prob])
        h_feature[self.feature_name_pre + 'SfEntropy'] = link_entropy
        return h_feature

    def _extract_per_e_prf(self, h_q_info, e):
        """

        :param h_q_info:
        :param e:
        :return: has_other, p_other
        """
        h_feature = dict()
        p = self._find_entity_p(h_q_info, e)
        ana = h_q_info[self.tagger]['query'][p]

        sf = h_q_info['query'][ana[1]:ana[2]]

        if sf not in self.h_surface_info:
            logging.warn('surface [%s] not found in dict', sf)
            return {}

        l_top_k = self.h_surface_info[sf]
        s_other = set([item for item, __ in l_top_k if item != e])

        e_cnt = 0
        other_cnt = 0

        l_rank_info = self.h_q_rank_info.get(h_q_info['qid'], [])
        for doc, score, h_info in l_rank_info[:self.prf_d]:
            l_ana = h_info.get(self.tagger, {}).get(body_field, [])
            l_e = [ana[0] for ana in l_ana]
            for this_e in l_e:
                if this_e == e:
                    e_cnt += 1
                if this_e in s_other:
                    other_cnt += 1

        h_feature[self.feature_name_pre + 'HasOtherLinkE'] = min(other_cnt, 1)
        h_feature[self.feature_name_pre + 'AppearedInPRF'] = min(e_cnt, 1)

        p_other = float(other_cnt) / max(1, float(other_cnt + e_cnt))
        h_feature[self.feature_name_pre + 'ProbLinkOther'] = p_other
        return h_feature










