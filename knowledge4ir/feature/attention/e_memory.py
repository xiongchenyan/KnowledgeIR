"""
e id feature
overfit to get ground truth term weights
"""

from knowledge4ir.feature.attention import (
    EntityAttentionFeature,
)
from traitlets import (
    Unicode,
)


class EntityMemoryAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode("EntityMem")

    def extract(self, h_q_info, l_e):
        l_h_feature = []
        for e in l_e:
            h = dict()
            h[self.feature_name_pre + e] = 1
            l_h_feature.append(h)

        return l_h_feature
