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
from knowledge4ir.joint import (
    SPOT_FIELD,
    COREFERENCE_FIELD
)
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    body_field,
    QUERY_FIELD,
    term2lm,
    mean_pool_feature,
    log_sum_feature,

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
    ana_format = Unicode('spot', help='annotation format, tagme or spot').tag(config=True)

    def __init__(self, **kwargs):
        super(AnaMatch, self).__init__(**kwargs)
        logging.info('ana match features uses [%s] annotation', self.ana_format)

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

    def _get_field_entity(self, h_info, field):
        if self.ana_format == 'spot':
            return self._get_spot_field_entity(h_info, field)
        else:
            return self._get_tagme_field_entity(h_info, field)

    def _get_spot_field_entity(self, h_info, field):
        l_ana = h_info.get(SPOT_FIELD, {}).get(field, [])
        l_e = []
        for ana in l_ana:
            e = ana['entities'][0]['id']
            l_e.append(e)
        return l_e

    def _get_tagme_field_entity(self, h_info, field):
        l_ana = h_info.get('tagme', {}).get(field, [])
        l_e = [ana[0] for ana in l_ana]
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
        # of different name variations (total only)
        # of clusters (total only)
    """
    feature_name_pre = Unicode('CoRef')

    def extract_pair(self, q_info, doc_info):
        """
        extract features using doc_infor's coreference field
        :param q_info:
        :param doc_info:
        :return: h_feature
        """
        l_q_e_id = [ana['entities'][0]['id'] for ana in q_info[SPOT_FIELD]['query']]

        l_h_stats = []
        for q_e_id in l_q_e_id:
            l_mentions = self._find_match_mentions(q_e_id, doc_info)
            h_stats = self._mention_stats(l_mentions)
            l_h_stats.append(h_stats)
        h_feature = self._pull_stats_to_features(l_h_stats)
        h_feature = dict([(self.feature_name_pre + key, value)
                          for key, value in h_feature.items()
                          ])
        return h_feature

    def _find_match_mentions(self, e_id, doc_info):
        """
        find matched mentions with e_id
        1: get all loc of e_id (in fields)
        2: find all mentions in coreferences that aligns e_id's location
            align == head in e_id's location and equal st
        :param e_id:
        :param doc_info:
        :return: l_mentions = [mentions of e_id in coreferences]
        """

        h_loc = self._get_e_location(e_id, doc_info)

        l_mentions = []
        for mention in doc_info[COREFERENCE_FIELD]:
            mention_cluster = mention['mentions']
            for p in xrange(len(mention_cluster)):
                if mention_cluster[p]['source'] == 'body':
                    mention_cluster[p]['source'] = body_field

            if self._mention_aligned(h_loc, mention_cluster):
                l_mentions.append(mention_cluster)

        return l_mentions

    @classmethod
    def _mention_stats(cls, l_mentions):
        h_stats = dict()
        h_stats['nb_mentions'] = len(l_mentions)

        h_field_cnt = dict(zip(TARGET_TEXT_FIELDS, [0] * TARGET_TEXT_FIELDS))
        s_name = set()
        for mention_cluster in l_mentions:
            for sf in mention_cluster:
                h_field_cnt[sf['source'] + '_cnt'] += 1
                s_name.add(sf['surface'])
        h_stats.update(h_field_cnt)
        h_stats['name_variants'] = len(s_name)
        return h_stats

    @classmethod
    def _pull_stats_to_features(cls, l_h_stats):
        """
        combina stats to features
            mean
            log product, with min -20
        :param l_h_stats:
        :return: h_feature
        """

        h_feature = dict()
        h_feature.update(mean_pool_feature(l_h_stats))
        h_feature.update(log_sum_feature(l_h_stats))
        return h_feature

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

    @classmethod
    def _mention_aligned(cls, h_loc, mention_cluster):
        """
        check if the mention cluster (coreferences) is aligned with h_loc
        alignment definition
            has a surface's head location in h_loc, and equal
        :param h_loc: field->st->ed
        :param mention_cluster: a mention cluster of coreferences,
        :return:
        """

        for sf in mention_cluster:
            field = sf['source']
            head_pos = sf['head']
            st = sf['loc'][0]
            if field in h_loc:
                if st in h_loc[field][st]:
                    if h_loc[field][st] > head_pos:
                        return True
        return False












