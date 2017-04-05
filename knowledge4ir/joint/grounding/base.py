"""
the base class for grounding features
input:
    a json dict of q or doc
        with information packed
            "spot"
            "query"
            "fields"
    the external resource
        surface form dict (multiple?)
        surface form link p
        e embedding
        for PRF: q's d's info
do:
    extract grounding features for surface form: link or not
    extract grounding features for entities: whether it is to link to the current one
"""

import json
import logging
from traitlets.config import Configurable
from traitlets import (
    List,
    Unicode,
    Int,
)
from knowledge4ir.joint.resource import JointSemanticResource
from knowledge4ir.utils import TARGET_TEXT_FIELDS
from copy import deepcopy
from scipy import stats
from knowledge4ir.joint import (
    GROUND_FIELD,
    SPOT_FIELD,
)
import numpy as np


class Grounder(Configurable):
    feature_pre = Unicode()
    l_target_fields = List(Unicode, default_value=['query'] + TARGET_TEXT_FIELDS
                           ).tag(config=True)

    def __init__(self, **kwargs):
        super(Grounder, self).__init__(**kwargs)
        self.resource = None  # must be set

    def set_resource(self, external_resource):
        self.resource = external_resource

    def extract(self, h_info):
        """
        extract and add features for surfaces and entities in h_info['spot']
        :param h_info: spot->field->[h_sf_info] h_sf_info['entities'] = [h_e_info]
        :return: packed into h_info
        """
        assert self.resource is not None

        if SPOT_FIELD not in h_info:
            logging.WARN('spot field not found in h_info')
            return h_info
        h_new_info = deepcopy(h_info)
        h_spotted_field = h_info[SPOT_FIELD]
        h_grounded_field = {}
        for field in self.l_target_fields:
            l_h_sf_info = h_spotted_field.get(field, [])

            l_h_sf_info_with_feature = []
            for h_sf_info in l_h_sf_info:
                h_sf_info['field'] = field
                h_sf_feature = self.extract_for_surface(h_sf_info, h_info)
                h_sf_info['f'] = h_sf_feature
                l_h_e_info = h_sf_info.get('entities', [])
                l_h_e_info_with_feature = []
                for h_e_info in l_h_e_info:
                    h_e_feature = self.extract_for_entity(h_e_info, h_sf_info, h_info)
                    h_e_info['f'] = h_e_feature
                    l_h_e_info_with_feature.append(h_e_info)
                h_sf_info['entities'] = l_h_e_info_with_feature
                l_h_sf_info_with_feature.append(h_sf_info)
            h_grounded_field[field] = l_h_sf_info_with_feature
        del h_new_info[SPOT_FIELD]
        h_new_info[GROUND_FIELD] = h_spotted_field
        return h_new_info

    def extract_for_surface(self, h_sf_info, h_info):
        """
        extract multiple features here
            sf's CMNS entropy
            sf's max CMNS - second max
            sf's coverage fraction over the text
            sf's len
            sf's lp (TODO)
        :param h_sf_info: {sf, st, ed, field}
        :param h_info: same as extract's h_info
        :return: h_feature, features for this sf
        """
        h_feature = {}

        h_feature.update(self._surface_cmns_features(h_sf_info))
        h_feature.update(self._surface_coverage_features(h_sf_info, h_info))
        h_feature.update(self._surface_lp(h_sf_info))

        logging.debug('sf [%s] feature %s', h_sf_info['surface'], json.dumps(h_feature))
        return h_feature

    def extract_for_entity(self, h_e_info, h_sf_info, h_info):
        """
        extract entity disambiguation features:
            cmns
            votes from other surface' #1 cmns entities (using embedding cosine)
                vote by:
                    max
                    mean
                    Bin_1 [0.9, 1), Bin_2 [0.7, 0.9)

        :param h_e_info: e info
        :param h_sf_info: sf info
        :param h_info: total data info
        :return: h_feature for this entity
        """
        h_feature = {}
        e_id = h_e_info['id']
        h_feature['e_cmns'] = h_e_info['cmns']

        h_feature.update(self._entity_embedding_vote(e_id, h_info))
        logging.debug('e [%s] feature %s', e_id, json.dumps(h_feature))
        return h_feature

    def _surface_cmns_features(self, h_sf_info):
        h_feature = {}

        l_e = h_sf_info.get('entities', {})
        l_cmns = [e_info.get('cmns', 0) for e_info in l_e]

        entropy = stats.entropy(l_cmns)

        l_cmns.sort(reverse=True)
        l_cmns.append(0)
        diff = l_cmns[0] - l_cmns[1]

        h_feature['sf_cmns_entropy'] = entropy
        h_feature['sf_cmns_topdiff'] = diff

        return h_feature

    def _surface_coverage_features(self, h_sf_info, h_info):
        h_feature = {}
        loc = h_sf_info['loc']
        field = h_sf_info['field']
        h_feature['sf_coverage'] = float(loc[1] - loc[0]) / len(h_info.get(field, "").split())
        h_feature['sf_len'] = len(h_sf_info.get('surface', ''))
        return h_feature

    def _surface_lp(self, h_sf_info):
        h_feature = {}
        sf = h_sf_info['surface']

        h_stat = self.resource.h_surface_stat.get(sf, {})
        wiki_tf = h_stat.get('tf', 0)
        lp = 0
        if wiki_tf >= 10:
            lp = h_stat.get('lp', 0)
        # h_feature['sf_wiki_tf'] = wiki_tf
        h_feature['sf_lp'] = lp
        return h_feature

    def _entity_embedding_vote(self, e_id, h_info):
        l_sim = []
        if e_id in self.resource.embedding:

            for field, l_sf in h_info['spot'].items():
                for sf in l_sf:
                    top_e_id = sf.get('entities', [{}])[0].get('id', '')
                    if top_e_id == e_id:
                        # no self vote
                        continue
                    if top_e_id not in self.resource.embedding:
                        continue
                    sim = self.resource.embedding.similarity(e_id, top_e_id)
                    l_sim.append(sim)

        max_sim, mean_sim, l_bin = self._pool_sim_score(l_sim)
        h_feature = dict()
        h_feature['e_vote_emb_max'] = max_sim
        h_feature['e_vote_emb_mean'] = mean_sim
        for i in xrange(len(l_bin)):
            h_feature['e_vote_bin_%d' % i] = l_bin[i]
        return h_feature

    @classmethod
    def _pool_sim_score(cls, l_sim, l_weight=None):
        max_sim = 0
        mean_sim = 0
        l_bin = [0, 0, 0]
        if not l_sim:
            return max_sim, mean_sim, l_bin
        max_sim = max(l_sim)
        if l_weight is None:
            l_weight = [1] * len(l_sim)
        s = np.array(l_sim)
        w = np.array(l_weight)
        mean_sim = s.dot(w) / sum(l_weight)
        for sim, weight in zip(l_sim, l_weight):
            if sim == 1:
                l_bin[0] += weight
            if 0.75 <= sim < 1:
                l_bin[1] += weight
            if 0.5 <= sim < 0.75:
                l_bin[2] += weight
        return max_sim, mean_sim, l_bin


