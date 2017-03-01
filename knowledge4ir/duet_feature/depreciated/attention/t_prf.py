"""
extract prf attention features for term
features:
    rm3 score (without idf)

    appear percentage in title in prf docs
    appear percentage in body in prf docs
prf docs:
    top 20 from base retrieval
"""


from knowledge4ir.duet_feature.attention import (
    TermAttentionFeature,
    mul_update,
)
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
    text2lm,
    rm3,
)


class TermPrfAttentionFeature(TermAttentionFeature):
    feature_name_pre = Unicode('Prf')
    prf_d = Int(20).tag(config=True)

    def __init__(self, **kwargs):
        super(TermPrfAttentionFeature, self).__init__(**kwargs)
        # self.h_field_h_df = {}
        # self.h_corpus_stat = {}
        self.h_q_rank_info = {}

    def set_external_info(self, external_info):
        super(TermPrfAttentionFeature, self).set_external_info(external_info)
        # self.h_field_h_df = external_info.h_field_h_df
        # self.h_corpus_stat = external_info.h_corpus_stat
        self.h_q_rank_info = dict(external_info.ll_q_rank_info)

    def extract(self, h_q_info, l_t):
        l_h_feature = {}

        h_field_l_doc_lm = self._form_prf_field_lm(h_q_info['qid'])
        prf_ranking = [item[:2] for item in self.h_q_rank_info.get(h_q_info['qid'], [])][:self.prf_d]
        l_h_feature = self.extract_prf(self.feature_name_pre, l_t, prf_ranking, h_field_l_doc_lm[body_field])
        l_h_feature = mul_update(l_h_feature, self.extract_coverage(self.feature_name_pre, l_t, h_field_l_doc_lm))
        return l_h_feature

    def _form_prf_field_lm(self, qid):
        l_rank_info = self.h_q_rank_info.get(qid, [])
        h_field_l_doc_lm = {}
        for field in TARGET_TEXT_FIELDS:
            l_doc_lm = []
            for doc, score, h_info in l_rank_info[:self.prf_d]:
                text = h_info.get(field, "")
                lm = text2lm(text, clean=True)
                l_doc_lm.append(lm)
            h_field_l_doc_lm[field] = l_doc_lm

        return h_field_l_doc_lm

    @classmethod
    def extract_prf(cls, feature_name_pre, l_t, prf_ranking, l_doc_lm):

        l_exp_terms = rm3(prf_ranking, l_doc_lm,
                          l_doc_h_df=None,
                          # total_df=self.h_corpus_stat[body_field]['total_df'],
                          # h_total_df=self.h_field_h_df[body_field]
                          )

        h_t_score = dict(zip(l_t, [0] * len(l_t)))
        h_t_rank = dict(zip(l_t, [0] * len(l_t)))

        for p in xrange(len(l_exp_terms)):
            t, score = l_exp_terms[p]
            if t in h_t_rank:
                h_t_rank[t] = 1.0 / p
                h_t_score[t] = score

        l_h_feature = []
        for t in l_t:
            er = h_t_rank[t]
            score = h_t_score[t]
            h_feature = {}
            h_feature[feature_name_pre + 'Err'] = er
            h_feature[feature_name_pre + 'Rm3'] = score
            l_h_feature.append(h_feature)

        return l_h_feature

    @classmethod
    def extract_coverage(cls, feature_name_pre, l_t, h_field_l_doc_lm):
        l_h_feature = []
        h_field_h_t_cnt = {}
        for field, l_doc_lm in h_field_l_doc_lm.items():
            h_field_h_t_cnt[field] = dict(zip(l_t, [0] * len(l_t)))
            for lm in l_doc_lm:
                for t in l_t:
                    if t in lm:
                        h_field_h_t_cnt[field][t] += 1

        for t in l_t:
            h_feature = dict()
            for field, h_t_cnt in h_field_h_t_cnt.items():
                h_feature[feature_name_pre + field.title() + 'Cov'] = h_t_cnt[t]
            l_h_feature.append(h_feature)

        return l_h_feature









