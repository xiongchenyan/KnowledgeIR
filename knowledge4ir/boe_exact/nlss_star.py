"""
star features from nlss

votes from qe's neighbors
star -> only edges from qe are considered
"""

import json
from traitlets import (
    Unicode,
    Int,
    List,
)
from knowledge4ir.boe_exact.nlss_feature import NLSSFeature
import logging
from knowledge4ir.utils import (
    body_field,
    title_field,
    add_feature_prefix,
    text2lm,
    term2lm,
    mean_pool_feature,
    sum_pool_feature,
)
from knowledge4ir.utils.boe import form_boe_per_field
from knowledge4ir.utils.retrieval_model import RetrievalModel


class NLSSStar(NLSSFeature):
    feature_name_pre = Unicode('NLSS_Star')
    l_target_fields = List(Unicode, default_value=[body_field]).tag(config=True)

    def __init__(self, **kwargs):
        super(NLSSStar, self).__init__(**kwargs)
        self.current_e = None  # updates whenever extracting for a new query entity
        self.l_this_nlss = []  # nlss of current e
        self.l_this_nlss_lm = []  # nlss lm of l_this_nlss
        self.h_e_nlss_idx = dict()  # tail e -> nlss p index, p is location in l_this_nlss

    def set_resource(self, resource):
        super(NLSSStar, self).set_resource(resource)
        if len(self.resource.l_h_nlss) > 1:
            logging.warn('NLSSStar only using first nlss set for now')

    def _construct_e_nlss_cash_info(self, l_this_nlss):
        """
        e -> [pos in l_this_nlss]
        :param l_this_nlss:
        :return:
        """
        self.l_this_nlss = l_this_nlss
        self.l_this_nlss_lm = [text2lm(sent, clean=True) for sent, __ in l_this_nlss]
        h_e = dict()
        for p in xrange(len(l_this_nlss)):
            l_e = l_this_nlss[p][1]
            for e in l_e:
                if e == self.current_e:
                    continue
                if e not in h_e:
                    h_e[e] = []
                h_e[e].append(p)
        self.h_e_nlss_idx = h_e

    def extract_per_entity(self, q_info, ana, doc_info):
        h_feature = dict()
        e_id = ana['id']
        if e_id != self.current_e:
            self.current_e = e_id
            self._construct_e_nlss_cash_info(self.resource.l_h_nlss[0])

        for field in self.l_target_fields:
            l_field_ana = form_boe_per_field(doc_info, field)
            h_field_lm = text2lm(doc_info.get(field, ""), clean=True)

            h_feature.update(add_feature_prefix(
                self._connected_emb_vote(l_field_ana),
                field))

            h_feature.update(add_feature_prefix(
                self._edge_cnt(l_field_ana),
                field))

            h_feature.update(add_feature_prefix(
                self._edge_retrieval(l_field_ana, h_field_lm, field),
                field))

        return h_feature

    def _connected_emb_vote(self, l_field_ana):
        h_feature = {}

        if self.current_e not in self.resource.embedding:
            h_feature['emb_vote'] = 0
            return h_feature

        l_e = [ana['id'] for ana in l_field_ana if ana['id'] in self.h_e_nlss_idx]
        vote_sum = 0
        for e in l_e:
            vote_score = self.resource.embedding.similarity(self.current_e, e)
            vote_sum += max(vote_score, 0)
        h_feature['emb_vote'] = vote_sum
        return h_feature

    def _edge_cnt(self, l_field_ana):
        h_feature = {}
        l_e = [ana['id'] for ana in l_field_ana if ana['id'] in self.h_e_nlss_idx]
        h_feature['edge_cnt'] = len(l_e)
        h_feature['uniq_tail'] = len(set(l_e))
        return h_feature

    def _edge_retrieval(self, l_field_ana, h_field_lm, field):
        """
        for each edge in this doc field
            get edge sent's lm
            calc retrieval scores
        sum up retrieval score to final feature
        :param l_field_ana:
        :param h_field_lm:
        :return:
        """
        h_feature = {}

        l_e = [ana['id'] for ana in l_field_ana if ana['id'] in self.h_e_nlss_idx]

        l_h_retrieval_scores = []
        l_h_avg_retrieval_scores = []
        h_e_tf = term2lm(l_e)
        for e, tf in h_e_tf.items():
            l_sent_lm = [self.l_this_nlss_lm[pos] for pos in self.h_e_nlss_idx[e]]
            l_this_e_h_scores = []
            for sent_lm in l_sent_lm:
                r_model = RetrievalModel()
                r_model.set_from_raw(
                    sent_lm, h_field_lm,
                    self.resource.corpus_stat.h_field_df.get(field, None),
                    self.resource.corpus_stat.h_field_total_df.get(field, None),
                    self.resource.corpus_stat.h_field_avg_len.get(field, None)
                )
                l_scores = r_model.scores()
                l_scores = [(name, v * tf) for name, v in l_scores]
                h_retrieval_score = dict(l_scores)
                l_h_retrieval_scores.append(h_retrieval_score)
                l_this_e_h_scores.append(h_retrieval_score)

            h_this_e_avg_score = mean_pool_feature(l_this_e_h_scores)
            l_h_avg_retrieval_scores.append(h_this_e_avg_score)
        h_feature.update(sum_pool_feature(l_h_retrieval_scores))
        h_feature.update(sum_pool_feature(l_h_avg_retrieval_scores))

        return h_feature





