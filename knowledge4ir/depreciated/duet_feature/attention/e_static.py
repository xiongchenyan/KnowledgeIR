"""
entity static features:
to describe the static importance of an entity
    has wiki link (need additional data)
    # of alias
    # of types (need additional data)
    log (desp length)
    # of neighbors (need additional data)
    has notable type (need additional data)
will need:
    the pre filtered triples for query entities


"""

from knowledge4ir.duet_feature.attention import (
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
import math
from knowledge4ir.utils.FbDumpBasic import FbDumpParser


class EntityStaticAttentionFeature(EntityAttentionFeature):
    feature_name_pre = Unicode("EntityStatic")
    
    def __init__(self, **kwargs):
        super(EntityStaticAttentionFeature, self).__init__(**kwargs)
        self.h_e_triples = {}

    def set_external_info(self, external_info):
        self.h_e_triples = external_info.h_e_triples

    def extract(self, h_q_info, l_e):
        # l_h_feature = []

        l_h_feature = self._extract_from_triples(l_e)

        return l_h_feature

    def _extract_from_triples(self, l_e):
        l_h_feature = []
        for e in l_e:
            l_h_feature.append(self._extract_per_e_with_triples(e))

        return l_h_feature

    def _extract_per_e_with_triples(self, e):
        h_feature = {}

        l_v_col = self.h_e_triples.get(e, [])
        parser = FbDumpParser()
        l_types = parser.get_type(l_v_col)
        is_person = 0
        is_location = 0
        for t in l_types:
            if 'people' in t:
                is_person = 1
            if 'location' in t:
                is_location = 1
        h_feature[self.feature_name_pre + 'IsLocation'] = is_location
        h_feature[self.feature_name_pre + 'isPerson'] = is_person

        # l_alias = parser.get_alias(l_v_col)
        # h_feature[self.feature_name_pre + 'NbAlias'] = math.log(max(len(l_alias), 1))

        # h_feature[self.feature_name_pre + 'NbTypes'] = math.log(max(len(l_types), 1))


        # desp = parser.get_desp(l_v_col)
        # h_feature[self.feature_name_pre + 'DespLen'] = math.log(max(len(desp.split()), 1))

        # notable = parser.get_notable(l_v_col)
        # has_notable = 0
        # if notable:
        #     has_notable = 1
        # h_feature[self.feature_name_pre + 'HasNotable'] = has_notable
        #
        # l_neighbors = parser.get_neighbor(l_v_col)
        # h_feature[self.feature_name_pre + 'NbNeighbor'] = math.log(max(len(l_neighbors), 1))
        #
        # h_feature[self.feature_name_pre + 'NbTriples'] = math.log(max(len(l_v_col), 1))

        return h_feature












