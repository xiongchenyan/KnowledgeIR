"""
attention feature base
base class with API defined, and data set
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List
)
import logging
import numpy as np


class TermAttentionFeature(Configurable):
    feature_name_pre = Unicode('TAtt')

    def set_external_info(self, external_info):
        logging.info('setting external info with shared storage')
        return

    def extract(self, h_q_info, l_t):
        yield NotImplementedError


class EntityAttentionFeature(Configurable):
    feature_name_pre = Unicode('EAtt')

    def set_external_info(self, external_info):
        logging.info('setting external info with shared storeage')
        return

    def extract(self, h_q_info, l_e):
        yield NotImplementedError

    def _find_entity_p(self, h_q_info, e):
        l_ana = h_q_info['tagme']['query']
        for p in xrange(len(l_ana)):
            if l_ana[p][0] == e:
                return p
        return -1


def form_avg_emb(l_node, emb):
    l_vector = [emb[node] for node in l_node if node in emb]
    if l_vector:
        return np.mean(np.array(l_vector), axis=0)
    return None


def calc_query_entity_total_embedding(h_q_info, emb):
    l_t = h_q_info['query'].lower().split()
    l_e = []
    for tagger in ['tagme', 'cmns']:
        if tagger in h_q_info:
            l_e.extend([ana[0] for ana in h_q_info[tagger]['query']])
    l_total = l_t + l_e
    return form_avg_emb(l_total, emb)


def mul_update(l_h_feature, l_this_h_feature):
    if not l_h_feature:
        l_h_feature = l_this_h_feature
    else:
        for p in xrange(len(l_h_feature)):
            l_h_feature[p].update(l_this_h_feature[p])
    return l_h_feature
