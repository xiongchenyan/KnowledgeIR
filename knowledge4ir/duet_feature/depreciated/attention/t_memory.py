"""
memory based features
    word name
"""

from knowledge4ir.duet_feature.attention import (
    TermAttentionFeature,
)
from traitlets import (
    Unicode,
)


class TermMemoryAttentionFeature(TermAttentionFeature):
    feature_name_pre = Unicode("TermMem")

    def extract(self, h_q_info, l_w):
        l_h_feature = []
        for w in l_w:
            h = dict()
            h[self.feature_name_pre + w.lower()] = 1
            l_h_feature.append(h)
        return l_h_feature

