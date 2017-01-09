"""

"""


from knowledge4ir.feature.attention import (
    EntityAttentionFeature,
    mul_update,
)
import json
import logging
from traitlets import (
    List,
    Unicode,
    Int
)


class EntitySurfaceFormAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode("EntitySurface")

    def extract(self, h_q_info, l_e):
        l_h_feature = []
        for e in l_e:
            l_h_feature.append(self._extract_surface_per_e(h_q_info, e))
        return l_h_feature

    def _extract_surface_per_e(self, h_q_info, e):
        h_feature = {}
        ana = []
        l_ana = h_q_info['tagme']['query']
        for p in l_ana:
            if l_ana[p][0] == e:
                ana = l_ana[p][0]
                break

        query = h_q_info['query']

        st, ed = ana[1:3]
        full_cover = 0
        if (ed - st) == len(query):
            full_cover = 1
        h_feature[self.feature_name_pre + 'FullCover'] = full_cover

        overlapped = 0
        for ana in l_ana:
            if e == ana[0]:
                continue
            if (ana[1] < ed) & (ana[2] >= ed):
                overlapped = 1
            if (ana[1] <= st) & (ana[2] > st):
                overlapped = 1

        h_feature[self.feature_name_pre + 'Overlapped'] = overlapped

        be_covered = 0
        for ana in l_ana:
            if (ana[1] <= st) & (ana[2] >= ed):
                be_covered = 1
        h_feature[self.feature_name_pre + 'BeCovered'] = be_covered
        h_feature[self.feature_name_pre + 'SoleE'] = min(1, len(l_ana))
        question_e = 0
        if query.split()[0].lower() in {'what', 'why', 'when', 'how'}:
            question_e = 1
        h_feature[self.feature_name_pre + 'QuestionE'] = question_e

        return h_feature





