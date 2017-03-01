"""
entity prf
the same as t_prf
"""

from traitlets import (
    Unicode,
    Int
)

from knowledge4ir.depreciated.duet_feature.attention.t_prf import TermPrfAttentionFeature
from knowledge4ir.duet_feature.attention import (
    EntityAttentionFeature,
    mul_update,
)
from knowledge4ir.utils import (
    body_field,
    TARGET_TEXT_FIELDS,
    term2lm,
    # rm3,
)


class EntityPrfAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode('Prf')
    prf_d = Int(20).tag(config=True)
    tagger = Unicode('tagme', help="tagger").tag(config=True)

    def __init__(self, **kwargs):
        super(EntityPrfAttentionFeature, self).__init__(**kwargs)
        # self.h_field_h_df = {}
        # self.h_corpus_stat = {}
        self.h_q_rank_info = {}

    def set_external_info(self, external_info):
        super(EntityPrfAttentionFeature, self).set_external_info(external_info)
        self.h_q_rank_info = dict(external_info.ll_q_rank_info)

    def extract(self, h_q_info, l_e):
        # l_h_feature = {}
        h_field_l_doc_lm = self._form_prf_field_lm(h_q_info['qid'])
        prf_ranking = [item[:2] for item in self.h_q_rank_info.get(h_q_info['qid'], [])][:self.prf_d]
        l_h_feature = TermPrfAttentionFeature.extract_prf(self.feature_name_pre, l_e, prf_ranking, h_field_l_doc_lm[body_field])
        l_h_feature = mul_update(l_h_feature, TermPrfAttentionFeature.extract_coverage(self.feature_name_pre, l_e, h_field_l_doc_lm))
        return l_h_feature

    def _form_prf_field_lm(self, qid):
        l_rank_info = self.h_q_rank_info.get(qid, [])
        h_field_l_doc_lm = {}
        for field in TARGET_TEXT_FIELDS:
            l_doc_lm = []
            for doc, score, h_info in l_rank_info[:self.prf_d]:
                l_ana = h_info.get(self.tagger, {}).get(field, [])
                l_e = [ana[0] for ana in l_ana]
                lm = term2lm(l_e)
                l_doc_lm.append(lm)
            h_field_l_doc_lm[field] = l_doc_lm

        return h_field_l_doc_lm

