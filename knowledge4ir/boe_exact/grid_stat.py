"""
count grid stat

count:
    # of grids
    avg # of entity per grid
    # number of entities that connected by qe's nlkg
"""


import json
import logging

import numpy as np
from scipy.spatial.distance import cosine
from traitlets import (
    Int,
    Unicode,
    List,
)

from knowledge4ir.boe_exact.boe_feature import BoeFeature
from knowledge4ir.utils import (
    mean_pool_feature,
    QUERY_FIELD,
    add_feature_prefix,
    text2lm,
    avg_embedding,
    lm_cosine,
    TARGET_TEXT_FIELDS,
    body_field,
    E_GRID_FIELD,
)
from knowledge4ir.utils.boe import (
    form_boe_per_field,
)
from knowledge4ir.utils.retrieval_model import RetrievalModel
from knowledge4ir.utils.nlp import text2lm
from knowledge4ir.boe_exact.nlss_feature import NLSSFeature


class GridStat(NLSSFeature):
    feature_name_pre = Unicode('grid_stat')

    def set_resource(self, resource):
        self.resource = resource
        assert self.resource.l_h_nlss

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_this_nlss):
        """
        # of grid edge
        # of e per edge
        # of e connected by qe's nlss
        :param q_info:
        :param ana:
        :param doc_info:
        :param l_this_nlss:
        :return:
        """

        nb_grid_edge = 0
        avg_e_per_grid = 0
        avg_e_connected = 0

        s_connected_e = set(
            sum([nlss[1] for nlss in l_this_nlss],
                [])
        )
        logging.info('[%s] has [%d] connected nlkg e', len(s_connected_e))
        qe = ana['id']
        l_grids = doc_info.get(E_GRID_FIELD, {}).get(body_field, [])
        for grid in l_grids:
            l_grid_ana = grid['spot']
            l_grid_e = [ana['id'] for ana in l_grid_ana]
            s_grid_e = set(l_grid_e)
            if qe in s_grid_e:
                nb_grid_edge += 1
                avg_e_per_grid += len(l_grid_e)
                avg_e_connected = len([e for e in l_grid_e if e in s_connected_e])

        if nb_grid_edge:
            avg_e_connected /= nb_grid_edge
            avg_e_per_grid /= nb_grid_edge
        h_feature = {
            "nb_grids": nb_grid_edge,
            'avg_e_per_grid': avg_e_per_grid,
            'avg_e_connected': avg_e_connected
        }
        return h_feature

