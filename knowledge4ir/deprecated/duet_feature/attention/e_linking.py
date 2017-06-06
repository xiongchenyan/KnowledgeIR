"""
entity linking features:
lp
link score
linker confidences:
    name has ()
"""


from knowledge4ir.duet_feature.attention import (
    EntityAttentionFeature,
    mul_update,
)
from traitlets import (
    Unicode,
)


class EntityLinkerAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode("EntityLinker")

    def extract(self, h_q_info, l_e):
        l_h_feature = []
        for e in l_e:
            l_h_feature.append(self._extract_linker_per_e(h_q_info, e))
        return l_h_feature

    def _extract_linker_per_e(self, h_q_info, e):
        h_feature = dict()
        p = self._find_entity_p(h_q_info, e)
        ana = h_q_info['tagme']['query'][p]

        h = ana[3]

        # linked prob and linking score
        for key, score in h.items():
            h_feature[self.feature_name_pre + key.title()] = score

        name = ana[-1]
        has_bracket = 0
        if ('(' in name) & (')' in name):
            has_bracket = 1
        h_feature[self.feature_name_pre + 'HasBracket'] = has_bracket

        return h_feature







