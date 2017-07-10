"""
star features from nlss

votes from qe's neighbors
star -> only edges from qe are considered
"""

import json
import logging

from traitlets import (
    Unicode,
    List,
)

from knowledge4ir.boe_exact.nlss_feature import NLSSFeature
from knowledge4ir.utils import (
    body_field,
    add_feature_prefix,
    text2lm,
    term2lm,
    mean_pool_feature,
    sum_pool_feature,
    QUERY_FIELD,
    E_GRID_FIELD,
)
from knowledge4ir.utils.boe import form_boe_per_field
from knowledge4ir.utils.retrieval_model import RetrievalModel


class NLSSStar(NLSSFeature):
    feature_name_pre = Unicode('NLSS_Star')
    l_target_fields = List(Unicode, default_value=[body_field]).tag(config=True)
    l_features = List(Unicode, default_value=['emb_vote', 'edge_cnt', 'edge_retrieval'],
                      help='nlss star features: emb_vote, qe_grid, nlss_grid'
                           'edge_cnt, edge_retrieval, local_grid, local_vote,'
                           'ltr_base'
                      ).tag(config=True)

    def __init__(self, **kwargs):
        super(NLSSStar, self).__init__(**kwargs)
        self.current_qid = None  # updates whenever extracting for a new query entity
        self.ll_this_nlss = []  # nlss of current ;l_qe
        self.h_qe_idx = dict()   # p of qe in ll_this_nlss, ll_this_nlss_lm, l_h_e_nlss_idx
        self.ll_this_nlss_lm = []  # nlss lm of l_this_nlss
        self.l_h_e_nlss_idx = dict()  # tail e -> nlss p index, p is location in l_this_nlss

    def set_resource(self, resource):
        super(NLSSStar, self).set_resource(resource)
        if len(self.resource.l_h_nlss) > 1:
            logging.warn('NLSSStar only using first nlss set for now')

    def _construct_e_nlss_cash_info(self, q_info, h_nlss):
        """
        e -> [pos in l_this_nlss]
        :return:
        """
        logging.info('constructing nlss cash for q [%s]', q_info['qid'])
        l_q_ana = form_boe_per_field(q_info, QUERY_FIELD)
        l_qe = list(set([ana['id'] for ana in l_q_ana]))
        self.h_qe_idx = dict(zip(l_qe, range(len(l_qe))))
        self.ll_this_nlss = []
        self.ll_this_nlss_lm = []
        self.l_h_e_nlss_idx = []
        for qe in l_qe:
            logging.info('forming nlss cash for qe [%s]', qe)
            l_this_nlss = h_nlss.get(qe, [])
            l_this_nlss_lm = [text2lm(sent, clean=True) for sent, __ in l_this_nlss]
            h_e = dict()
            for p in xrange(len(l_this_nlss)):
                l_e = l_this_nlss[p][1]
                for e in l_e:
                    if e in qe:
                        continue
                    if e not in h_e:
                        h_e[e] = []
                    h_e[e].append(p)
            logging.info('qe [%s] [%d] nlss, [%d] tail e', qe, len(l_this_nlss), len(h_e))
            self.ll_this_nlss.append(l_this_nlss)
            self.ll_this_nlss_lm.append(l_this_nlss_lm)
            self.l_h_e_nlss_idx.append(h_e)
        logging.info('q [%s] nlss cash constructed', q_info['qid'])

    def extract_per_entity(self, q_info, ana, doc_info):
        h_feature = dict()
        qe = ana['id']
        qid = q_info['qid']
        logging.info('start extracting [%s]-[%s]-[%s]',
                     qid, qe, doc_info['docno'])
        if qid != self.current_qid:
            self.current_qid = qid
            self._construct_e_nlss_cash_info(q_info, self.resource.l_h_nlss[0])
        for field in self.l_target_fields:
            l_field_ana = form_boe_per_field(doc_info, field)
            h_field_lm = text2lm(doc_info.get(field, ""), clean=True)
            if 'emb_vote' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._connected_emb_vote(qe, l_field_ana),
                    field + '_'))
            if 'edge_cnt' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._edge_cnt(qe, l_field_ana),
                    field + '_'))
            if 'edge_retrieval' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._edge_retrieval(qe, l_field_ana, h_field_lm, field),
                    field + '_'))
            if 'local_grid' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._local_grid(q_info, qe, l_field_ana, doc_info, field),
                    field + '_'))
            if 'qe_grid' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._qe_grid(q_info, qe, doc_info, field),
                    field + '_'))
            if 'nlss_grid' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._nlss_grid(q_info, qe, l_field_ana, doc_info, field),
                    field + '_'))
            if 'ltr_base' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._ltr_baseline(q_info, h_field_lm, field),
                    field + '_'))
            if 'local_vote' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._local_vote(q_info, qe, l_field_ana, doc_info, field),
                    field + '_'
                ))
            if 'grid_retrieval' in self.l_features:
                h_feature.update(add_feature_prefix(
                    self._grid_retrieval(qe, h_field_lm, doc_info, field),
                    field + '_'
                ))

        return h_feature

    def _connected_emb_vote(self, qe, l_field_ana):
        h_feature = {}
        p = self.h_qe_idx[qe]
        h_e_nlss_idx = self.l_h_e_nlss_idx[p]
        z = max(float(len(l_field_ana)), 1.0)
        if qe not in self.resource.embedding:
            h_feature['emb_vote'] = 0
            return h_feature

        l_de = [ana['id'] for ana in l_field_ana if ana['id'] in h_e_nlss_idx]
        logging.info('qe [%s] has [%d] connected de', qe, len(l_de))
        vote_sum = 0
        for de in l_de:
            if de not in self.resource.embedding:
                continue
            vote_score = self.resource.embedding.similarity(qe, de)
            vote_sum += max(vote_score, 0)
        h_feature['emb_vote'] = vote_sum / z
        return h_feature

    def _edge_cnt(self, qe, l_field_ana):
        h_feature = {}
        z = max(float(len(l_field_ana)), 1.0)
        h_e_nlss_idx = self.l_h_e_nlss_idx[self.h_qe_idx[qe]]
        l_e = [ana['id'] for ana in l_field_ana if ana['id'] in h_e_nlss_idx]
        h_feature['edge_cnt'] = len(l_e) / z
        h_feature['uniq_tail'] = len(set(l_e)) / z
        logging.info('qe [%s] edge cnt %s', qe, json.dumps(h_feature))
        return h_feature

    def _edge_retrieval(self, qe, l_field_ana, h_field_lm, field):
        """
        for each edge in this doc field
            get edge sent's lm
            calc retrieval scores
        sum up retrieval score to final feature
        :param qe:
        :param l_field_ana:
        :param h_field_lm:
        :return:
        """
        z = max(float(len(l_field_ana)), 1.0)
        h_feature = {}
        p = self.h_qe_idx[qe]
        h_e_nlss_idx = self.l_h_e_nlss_idx[p]
        l_this_nlss_lm = self.ll_this_nlss_lm[p]
        l_e = [ana['id'] for ana in l_field_ana if ana['id'] in h_e_nlss_idx]

        l_h_retrieval_scores = []
        l_h_avg_retrieval_scores = []
        h_e_tf = term2lm(l_e)
        avg_sent_per_e = 0
        for e, tf in h_e_tf.items():
            l_sent_lm = [l_this_nlss_lm[pos] for pos in h_e_nlss_idx[e]]
            avg_sent_per_e += len(l_sent_lm)
            l_this_e_h_scores = []
            for sent_lm in l_sent_lm:
                l_scores = self._extract_retrieval_scores(sent_lm, h_field_lm, field)
                l_scores = [(name, v * tf / z) for name, v in l_scores if 'lm_twoway' not in name]
                h_retrieval_score = dict(l_scores)
                l_h_retrieval_scores.append(h_retrieval_score)
                l_this_e_h_scores.append(h_retrieval_score)

            h_this_e_avg_score = mean_pool_feature(l_this_e_h_scores)
            l_h_avg_retrieval_scores.append(h_this_e_avg_score)
        avg_sent_per_e /= float(max(len(h_e_tf), 1.0))
        avg_sent_per_e = max(avg_sent_per_e, 1.0)
        h_sum_retrieval_score = sum_pool_feature(l_h_retrieval_scores)
        h_sum_retrieval_score = dict([(k, v / avg_sent_per_e)
                                      for k, v in h_sum_retrieval_score.items()])
        h_feature.update(h_sum_retrieval_score)
        h_feature.update(sum_pool_feature(l_h_avg_retrieval_scores))

        """
        make sure not too small values
        """
        h_feature = dict([(k, max(v, -100)) for k, v in h_feature.items()])
        return h_feature

    def _local_grid(self, q_info, qe, l_field_ana, doc_info, field):
        """
        only keep grids that
            1) include qe
            2) include qe->nlss->tail e
        :param q_info: query info
        :param qe:
        :param doc_info:
        :param field:
        :return:
        """
        p = self.h_qe_idx[qe]
        h_e_nlss_idx = self.l_h_e_nlss_idx[p]
        l_tail_e = [ana['id'] for ana in l_field_ana if ana['id'] in h_e_nlss_idx]

        l_qe_grid = []
        l_nlss_e_grid = []

        l_grid = doc_info.get(E_GRID_FIELD, {}).get(field, [])
        for grid in l_grid:
            l_grid_e = [ana['id'] for ana in grid['spot']]
            s_grid_e = set(l_grid_e)
            if qe in s_grid_e:
                l_qe_grid.append(grid['sent'])
            for tail_e in l_tail_e:
                if tail_e in s_grid_e:
                    l_nlss_e_grid.append(grid['sent'])
                    break
        logging.info('q [%s] e [%s] doc [%s] has [%d] qe grid, [%d] nlss grid',
                     q_info['qid'], qe, doc_info['docno'], len(l_qe_grid), len(l_nlss_e_grid)
                     )
        qe_grid_lm = text2lm(' '.join(l_qe_grid), clean=True)
        nlss_e_grid_lm = text2lm(' '.join(l_nlss_e_grid), clean=True)
        q_lm = text2lm(q_info[QUERY_FIELD])
        h_feature = {}

        h_qe_grid_scores = dict(self._extract_retrieval_scores(q_lm, qe_grid_lm, field))
        h_nlss_grid_scores = dict(self._extract_retrieval_scores(q_lm, nlss_e_grid_lm, field))

        h_feature.update(add_feature_prefix(h_qe_grid_scores, 'QEGrid_'))
        h_feature.update(add_feature_prefix(h_nlss_grid_scores, 'NlssGrid_'))
        return h_feature

    def _qe_grid(self, q_info, qe, doc_info, field):
        p = self.h_qe_idx[qe]
        h_e_nlss_idx = self.l_h_e_nlss_idx[p]

        l_qe_grid = []

        l_grid = doc_info.get(E_GRID_FIELD, {}).get(field, [])
        for grid in l_grid:
            l_grid_e = [ana['id'] for ana in grid['spot']]
            s_grid_e = set(l_grid_e)
            if qe in s_grid_e:
                l_qe_grid.append(grid['sent'])
        logging.info('q [%s] e [%s] doc [%s] has [%d] qe grid',
                     q_info['qid'], qe, doc_info['docno'], len(l_qe_grid)
                     )
        qe_grid_lm = text2lm(' '.join(l_qe_grid), clean=True)
        q_lm = text2lm(q_info[QUERY_FIELD])
        h_feature = {}
        h_qe_grid_scores = dict(self._extract_retrieval_scores(q_lm, qe_grid_lm, field))
        h_feature.update(add_feature_prefix(h_qe_grid_scores, 'QEGrid_'))
        return h_feature

    def _nlss_grid(self, q_info, qe, l_field_ana, doc_info, field):
        """
        only keep grids that
            1) include qe
            2) include qe->nlss->tail e
        :param q_info: query info
        :param qe:
        :param doc_info:
        :param field:
        :return:
        """
        p = self.h_qe_idx[qe]
        h_e_nlss_idx = self.l_h_e_nlss_idx[p]
        l_tail_e = [ana['id'] for ana in l_field_ana if ana['id'] in h_e_nlss_idx]

        l_nlss_e_grid = []

        l_grid = doc_info.get(E_GRID_FIELD, {}).get(field, [])
        for grid in l_grid:
            l_grid_e = [ana['id'] for ana in grid['spot']]
            s_grid_e = set(l_grid_e)
            for tail_e in l_tail_e:
                if tail_e in s_grid_e:
                    l_nlss_e_grid.append(grid['sent'])
                    break
        logging.info('q [%s] e [%s] doc [%s] has [%d] nlss grid',
                     q_info['qid'], qe, doc_info['docno'],  len(l_nlss_e_grid)
                     )
        nlss_e_grid_lm = text2lm(' '.join(l_nlss_e_grid), clean=True)
        q_lm = text2lm(q_info[QUERY_FIELD])
        h_feature = {}

        h_nlss_grid_scores = dict(self._extract_retrieval_scores(q_lm, nlss_e_grid_lm, field))

        h_feature.update(add_feature_prefix(h_nlss_grid_scores, 'NlssGrid_'))
        return h_feature

    def _ltr_baseline(self, q_info, h_field_lm, field):
        q_lm = text2lm(q_info[QUERY_FIELD])
        l_scores = self._extract_retrieval_scores(q_lm, h_field_lm, field)
        h_feature = dict(l_scores)
        return h_feature

    def _local_vote(self, q_info, qe, l_field_ana, doc_info, field):
        h_feature = {}
        if qe not in self.resource.embedding:
            h_feature['grid_emb_vote'] = 0
            return h_feature
        l_grid = doc_info.get(E_GRID_FIELD, {}).get(field, [])
        emb_vote = 0
        l_uw_e = []
        z = max(float(len(l_field_ana)), 1.0)
        for grid in l_grid:
            l_grid_e = [ana['id'] for ana in grid['spot']]
            s_grid_e = set(l_grid_e)
            if qe not in s_grid_e:
                continue
            for e in l_grid_e:
                if e == qe:
                    continue
                l_uw_e.append(e)
                if e not in self.resource.embedding:
                    continue
                emb_vote += self.resource.embedding.similarity(qe, e)

        h_feature['grid_emb_vote'] = emb_vote / z
        h_feature['grid_edge_cnt'] = len(l_uw_e) / z
        h_feature['grid_uniq_tail'] = len(set(l_uw_e)) / z
        return h_feature

    def _grid_retrieval(self, qe, h_field_lm, doc_info, field):
        l_grid = doc_info.get(E_GRID_FIELD, {}).get(field, [])
        z = float(len(l_grid))
        l_h_scores = []
        for grid in l_grid:
            l_grid_e = [ana['id'] for ana in grid['spot']]
            s_grid_e = set(l_grid_e)
            if qe not in s_grid_e:
                continue
            sent_lm = text2lm(grid['sent'], clean=True)
            l_scores = self._extract_retrieval_scores(sent_lm, h_field_lm, field)
            h_scores = dict([(k, v / z) for k, v in l_scores])
            l_h_scores.append(h_scores)
        h_feature = sum_pool_feature(l_h_scores)
        h_feature = add_feature_prefix(h_feature, 'grid_retrieval')
        return h_feature



