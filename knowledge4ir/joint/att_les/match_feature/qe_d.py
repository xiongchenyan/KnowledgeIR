"""
match q entities to d text

q e's texts as queries, extract e-d extraction features scores
one for each sf (st, ed)-> e candidate pair
    format:
        h_json: {match: {surface:"", loc: (), entities: {id:"", f:{feature name:score} }}
"""

import json
import logging

from traitlets.config import Configurable
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    ENTITY_TEXT_FIELDS,
    e_name_field,
    text2lm,
    GROUND_FIELD, SPOT_FIELD, MATCH_FIELD)
from traitlets import (
    Unicode,
    Set,
    List
)
from knowledge4ir.utils.retrieval_model import RetrievalModel
from knowledge4ir.joint import (
    GROUND_FIELD,
    SPOT_FIELD,
    MATCH_FIELD
)


class QeDTextMatchFeatureExtractor(Configurable):
    feature_name_pre = Unicode('QeD')
    l_entity_fields = List(Unicode, default_value=ENTITY_TEXT_FIELDS).tag(config=True)

    def __init__(self, **kwargs):
        super(QeDTextMatchFeatureExtractor, self).__init__(**kwargs)
        logging.info('initializing QeDTextMatchFeatureExtractor')
        self.retrieval_model = RetrievalModel(**kwargs)
        logging.info('QeDTextMatchFeatureExtractor init with target entity fields: %s',
                     json.dumps(self.l_entity_fields))

    @classmethod
    def class_print_help(cls, inst=None):
        super(QeDTextMatchFeatureExtractor, cls).class_print_help(inst)
        RetrievalModel.class_print_help(inst)

    def extract(self, q_info, d_info, external_resource):
        """

        :param q_info: grounded query info
        :param d_info: doc with textual fields
        :param external_resource: make sure h_entity_fields is loaded
        :return: matching features for each entities in the grounded fields, in same tree structure
        h_match_info:
            qid:
            docno:
            match: tree structure for each entities in grounded field
        """
        assert external_resource.h_entity_fields is not None

        h_match_info = dict()
        h_match_info['qid'] = q_info['qid']
        h_match_info['docno'] = d_info['docno']
        l_q_grounded = q_info[GROUND_FIELD]['query']
        l_q_matched_feature = []
        for grounded_sf in l_q_grounded:
            matched_sf = dict()
            matched_sf['surface'] = grounded_sf['surface']
            matched_sf['loc'] = grounded_sf['loc']
            l_matched_entities = []
            for grounded_e in grounded_sf['entities']:
                e_id = grounded_e['id']
                e_name = external_resource.h_entity_fields.get(e_id, {}).get(e_name_field, "")
                h_feature = self._extract_per_entity(e_id, d_info, external_resource)
                l_matched_entities.append(({'id': e_id, 'f': h_feature, e_name_field: e_name}))
            matched_sf['entities'] = l_matched_entities
            l_q_matched_feature.append(matched_sf)
        h_match_info[MATCH_FIELD] = l_q_matched_feature
        return h_match_info

    def _extract_per_entity(self, e_id, d_info, external_resource):
        h_feature = dict()
        h_e_fields = external_resource.h_entity_fields.get(e_id, {})
        l_e_text_fields = [(field, h_e_fields.get(field, ""))
                           for field in self.l_entity_fields]
        corpus_stat = external_resource.corpus_stat
        for field, text in l_e_text_fields:
            h_q_lm = text2lm(text, clean=True)
            for doc_field in TARGET_TEXT_FIELDS:
                doc_text = d_info.get(doc_field, "")
                h_d_lm = text2lm(doc_text, clean=True)
                self.retrieval_model.set(h_q_lm, h_d_lm, doc_field, corpus_stat)
                l_sim_scores = self.retrieval_model.scores()

                l_feature = [(self.feature_name_pre + field.title()
                              + doc_field.title() + name, score)
                             for name, score in l_sim_scores]
                h_feature.update(dict(l_feature))

        return h_feature









