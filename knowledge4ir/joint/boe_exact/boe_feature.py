"""
EF and Coor match feature
input:
    q info
    doc info
output:
    h_feature
"""

import json
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode,
    Bool
)
import logging
from knowledge4ir.joint import SPOT_FIELD
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    QUERY_FIELD,
    term2lm
)
from knowledge4ir.utils.retrieval_model import (
    RetrievalModel,
)


class BoeFeature(Configurable):
    feature_name_pre = Unicode()

    def extract_pair(self, q_info, doc_info):
        raise NotImplementedError


class AnaMatch(BoeFeature):
    feature_name_pre = Unicode('AnaMatch')

    def extract_pair(self, q_info, doc_info):
        """

        :param q_info: will use spot->query
        :param doc_info: will use spot -> doc
        only the first entity is used
        :return: h_feature={feature name : score}
        """

        l_q_e = self._get_q_entity(q_info)
        l_field_doc_e = self._get_doc_entity(doc_info)

        h_feature = dict()
        for field, l_e in l_field_doc_e:
            l_name_score = self._match_qe_de(l_q_e, l_e)
            for name, score in l_name_score:
                h_feature[self.feature_name_pre + name] = score

        return h_feature

    def _get_q_entity(self, q_info):
        return self._get_field_entity(q_info, QUERY_FIELD)

    def _get_doc_entity(self, doc_info):
        l_field_doc_e = [(field, self._get_field_entity(doc_info, field)) for field in TARGET_TEXT_FIELDS]
        return l_field_doc_e

    @classmethod
    def _get_field_entity(cls, h_info, field):
        l_ana = h_info[SPOT_FIELD][field]
        l_e = []
        for ana in l_ana:
            e = ana['entities'][0]['id']
            l_e.append(e)
        return l_e

    @classmethod
    def _match_qe_de(cls, l_qe, l_de):
        q_lm = term2lm(l_qe)
        d_lm = term2lm(l_de)
        retrieval_model = RetrievalModel()
        retrieval_model.set_from_raw(q_lm, d_lm)
        l_sim = list()
        l_sim.append(['tf', retrieval_model.tf()])
        l_sim.append(['lm', retrieval_model.lm()])
        l_sim.append(['coor', retrieval_model.coordinate()])
        l_sim.append(['bool_and', retrieval_model.bool_and()])
        return l_sim


class CoreferenceMatch(BoeFeature):
    """
    coreference features
    06/12/2017 version includes:
        has coreference in fields
        # of coreferences in fields
        # of different name variations (body only)
        # of clusters (body only)
    """

    def extract_pair(self, q_info, doc_info):
        return

    def _find_match_mentions(self, e_id, doc_info):
        """
        find matched mentions with e_id
        1: get all loc of e_id (in fields)
        2: find all mentions in coreferences that aligns e_id's location
            align == equal st, head in e_id's location
        :param e_id:
        :param doc_info:
        :return: l_mentions = [mentions of e_id in coreferences]
        """

        h_loc = self._get_e_location(e_id, doc_info)

        l_mentions = []



    @classmethod
    def _get_e_location(cls, e_id, doc_info):
        """
        find location of e_id
        :param e_id: target
        :param doc_info: spotted and coreferenced document
        :return: h_loc field-> st -> ed
        """
        h_loc = dict()
        for field in TARGET_TEXT_FIELDS:
            l_ana = doc_info[SPOT_FIELD][field]
            h_loc[field] = dict()
            for ana in l_ana:
                e = ana['entities'][0]['id']
                if e == e_id:
                    st, ed = ana['loc']
                    h_loc[field][st] = ed
        return h_loc









