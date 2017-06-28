"""
simple salient features, except frequency

version 1 (06282017)
position,
uw vote from w and from e
    pooled to one score
"""

from knowledge4ir.utils.boe import (
    uw_word_embedding_vote,
    entity_embedding_vote,
    word_embedding_vote,
    form_boe_per_field,
)
import json
import logging
import math
from knowledge4ir.boe_exact.boe_feature import BoeFeature
from traitlets import (
    Int,
    Unicode,
    Bool
)
from knowledge4ir.utils import (
    log_sum_feature,
    mean_pool_feature,
    max_pool_feature,
    QUERY_FIELD,
    TARGET_TEXT_FIELDS,
)


class SalientFeature(BoeFeature):
    feature_name_pre = Unicode('Salient')

    def set_resource(self, resource):
        self.resource = resource
        assert resource.embedding
        logging.info('Salient feature resource set')

    def extract_pair(self, q_info, doc_info):
        logging.debug('extracting salient features for [%s][%s]',
                      q_info['qid'], doc_info['docno'])
        l_q_ana = self._get_field_ana(q_info, QUERY_FIELD)

        l_h_feature = [self.extract_per_entity(ana, doc_info) for ana in l_q_ana]

        h_final_feature = {}
        h_final_feature.update(log_sum_feature(l_h_feature))
        h_final_feature.update(mean_pool_feature(l_h_feature))
        h_final_feature = dict([(self.feature_name_pre + item[0], item[1])
                                for item in h_final_feature.items()])

        return h_final_feature

    def extract_per_entity(self, ana, doc_info):
        """
        extract for one ana in the q_info
        current features:
            first position (log(loc))
            uw word vote, average and max
            e vote
            nb of e with > 0.2 embedding similarity
        :param ana:
        :param doc_info:
        :return: the features for this ana in doc_info
        """

        h_feature = dict()
        h_loc = self._get_e_location(ana['id'], doc_info)

        h_feature.update(self._extract_w_votes(ana, h_loc, doc_info))

        h_feature.update(self._extract_e_votes(ana, h_loc, doc_info))

        h_feature.update(self._extract_pos(ana, h_loc, doc_info))
        logging.debug('[%s] feature %s', ana['id'], json.dumps(h_feature))
        return h_feature

    def _extract_w_votes(self, ana, h_loc, doc_info):
        """
        extract features from word's voting
            uw max
            uw average
            full average
        :param ana:
        :param h_loc:
        :param doc_info:
        :return:
        """
        h_feature = dict()
        e_id = ana['id']
        for field in TARGET_TEXT_FIELDS:
            l_h_uw_votes = []
            for loc in h_loc[field].items():
                h = uw_word_embedding_vote(e_id, doc_info, field, loc, self.resource)
                l_h_uw_votes.append(h)

            h_uw_avg = mean_pool_feature(l_h_uw_votes)
            h_uw_max = max_pool_feature(l_h_uw_votes)

            h_full_vote = word_embedding_vote(e_id, doc_info, field, self.resource)
            h_res = dict([(field + '_' + k, v)
                          for k, v in h_uw_avg.items() + h_uw_max.items() + h_full_vote.items()])
            h_feature.update(h_res)
        return h_feature

    def _extract_e_votes(self, ana, h_loc, doc_info):
        """
         extract features from word's voting
            full average
        :param ana:
        :param h_loc:
        :param doc_info:
        :return:
        """
        h_feature = dict()
        e_id = ana['id']
        for field in TARGET_TEXT_FIELDS:
            h_full_vote = entity_embedding_vote(e_id, doc_info, field, self.resource)
            h_res = dict([(field + '_' + k, v)
                          for k, v in h_full_vote.items()])
            h_feature.update(h_res)
        return h_feature

    def _extract_pos(self, ana, h_loc, doc_info):
        h_feature = {}
        for field in TARGET_TEXT_FIELDS:
            l_st = h_loc[field].keys()
            p = min(l_st)
            h_feature[field + '_FirstPos'] = math.log(p + 1.0)
        return h_feature



