"""
static features
I think it is only idf now
"""

from knowledge4ir.feature.attention import (
    TermAttentionFeature,
    mul_update,
)
import json
import logging
import math
from knowledge4ir.utils import body_field
from traitlets import (
    List,
    Unicode,
    Int
)


class TermStaticAttentionFeature(TermAttentionFeature):
    feature_name_pre = Unicode('Static')

    def __init__(self, **kwargs):
        super(TermStaticAttentionFeature, self).__init__(**kwargs)
        self.h_field_h_df = {}
        self.h_corpus_stat = {}

    def set_external_info(self, external_info):
        super(TermStaticAttentionFeature, self).set_external_info(external_info)
        self.h_field_h_df = external_info.h_field_h_df
        self.h_corpus_stat = external_info.h_corpus_stat

    def extract(self, h_q_info, l_t):
        l_h_feature = {}
        l_h_feature = self._extract_idf(l_t)
        return l_h_feature

    def _extract_idf(self, l_t):
        l_h_feature = []
        for t in l_t:
            h_feature = {}
            for field, h_df in self.h_field_h_df.items():
                if field != body_field:
                    continue
                df = h_df.get(t, 1)
                total_df = self.h_corpus_stat[field]['total_df']
                idf = math.log(float(total_df) / float(df))
                h_feature[self.feature_name_pre + field.title() + 'Idf'] = idf
            l_h_feature.append(h_feature)
        return l_h_feature




