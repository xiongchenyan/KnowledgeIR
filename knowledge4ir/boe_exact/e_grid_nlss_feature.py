"""
extract features using entity grid and nlss
focus on BOE now so only query entities are considered

starting with star model
    information from
        q e's sentences in doc VS q e's NLSS
        q e and other doc e's connection in doc VS their connection in q e's NLSS
    data prepared in:
        e_grid -> field (focus on body now)

features:
    node:
        # of grid sentence
        max * mean grid sentence's similarities with NLSS of qe (exact exact match)
            similarity = cosine bow + cosine avg embedding
        average across q entities
    edge:
        TBD

"""

from knowledge4ir.utils.boe import (
    form_boe_per_field,
)

from knowledge4ir.utils.retrieval_model import RetrievalModel
import json
import logging
import math
from knowledge4ir.boe_exact.boe_feature import BoeFeature
from traitlets import (
    Int,
    Unicode,
    Bool,
    List,
)
from knowledge4ir.utils import (
    log_sum_feature,
    mean_pool_feature,
    max_pool_feature,
    QUERY_FIELD,
    TARGET_TEXT_FIELDS,
    abstract_field,
    body_field,
    E_GRID_FIELD,
    add_feature_prefix,
    text2lm,
    avg_embedding,
    SPOT_FIELD,
)
import numpy as np


class EGridNLSSFeature(BoeFeature):
    """
    extract boe exact features by comparing e_grid of qe with qe's nlss
    """
    feature_name_pre = Unicode('EGridNLSS')
    l_target_fields = List(Unicode, default_value=[body_field]).tag(config=True)
    max_sent_len = Int(100, help='max grid sentence len to consider').tag(config=True)

    def set_resource(self, resource):
        self.resource = resource
        assert resource.embedding
        assert resource.l_h_nlss
        logging.info('Salient feature resource set')

    def extract_pair(self, q_info, doc_info):
        """

        :param q_info:
        :param doc_info:
        :return:
        """
        logging.debug('extracting e_grid nlss features for [%s][%s]',
                      q_info['qid'], doc_info['docno'])
        assert E_GRID_FIELD in doc_info
        l_q_ana = self._get_field_ana(q_info, QUERY_FIELD)
        logging.debug('q info %s', json.dumps(q_info))
        logging.debug('q ana %s', json.dumps(l_q_ana))
        logging.debug('doc t [%s], info [%s]', doc_info.get('title', ""),
                      json.dumps(doc_info.get('spot', {}).get('title', []))
                      )
        l_h_feature = [self.extract_per_entity(ana, doc_info) for ana in l_q_ana]

        h_final_feature = {}
        # h_final_feature.update(log_sum_feature(l_h_feature))
        h_final_feature.update(mean_pool_feature(l_h_feature))
        h_final_feature = dict([(self.feature_name_pre + item[0], item[1])
                                for item in h_final_feature.items()])

        return h_final_feature

    def extract_per_entity(self, ana, doc_info):
        """

        :param ana: one q ana
        :param doc_info:
        :return:
        """

        h_feature = dict()
        e_id = ana['id']
        ll_qe_nlss = [h_nlss.get(e_id, []) for h_nlss in self.resource.l_h_nlss]

        for nlss_name, l_qe_nlss in zip(self.resource.l_nlss_name, ll_qe_nlss):
            h_this_nlss_feature = self._extract_per_entity_via_nlss(ana, doc_info, l_qe_nlss)
            h_feature.update(add_feature_prefix(h_this_nlss_feature, nlss_name + '_'))
        return h_feature

    def _extract_per_entity_via_nlss(self, ana, doc_info, l_qe_nlss):
        """

        :param ana:
        :param doc_info:
        :param l_qe_nlss:
        :return:
        """

        h_this_feature = dict()
        h_e_grid = doc_info.get(E_GRID_FIELD, {})
        l_nlss_bow = self._form_nlss_bow(l_qe_nlss)
        l_nlss_emb = self._form_nlss_emb(l_qe_nlss)
        for field in self.l_target_fields:
            if field not in h_e_grid:
                continue
            l_e_grid = h_e_grid[field]
            h_field_grid_feature = self._extract_per_entity_per_nlss_per_field(
                ana, doc_info, l_qe_nlss, l_e_grid, l_nlss_bow, l_nlss_emb)
            h_this_feature.update(add_feature_prefix(h_field_grid_feature, field + '_'))
        return h_this_feature

    def _extract_per_entity_per_nlss_per_field(
            self, ana, doc_info, l_qe_nlss, l_e_grid,  l_nlss_bow, l_nlss_emb):
        """
        for each sentence in e_grid,
            check if ana e in it, and if len < max_sent_len
            calculate similarity with all qe_nlss
            average and max sum up
        :param ana:
        :param doc_info:
        :param l_qe_nlss: nlss of qe
        :param l_e_grid: grid of this field
        :param l_nlss_bow: pre calc bow of nlss
        :param l_nlss_emb: pre calc emb of nlss
        :return:
        """
        e_id = ana['id']
        l_this_e_grid = self._filter_e_grid(e_id, l_e_grid)
        l_grid_bow = self._form_grid_bow(l_this_e_grid)
        l_grid_emb = self._form_grid_emb(l_this_e_grid)

        m_bow_sim = self._calc_bow_trans(l_grid_bow, l_nlss_bow)
        m_emb_sim = self._calc_emb_trans(l_grid_emb, l_nlss_emb)

        h_bow_feature = self._pool_grid_nlss_sim(m_bow_sim)
        h_emb_feature = self._pool_grid_nlss_sim(m_emb_sim)

        h_feature = dict()
        h_feature.update(add_feature_prefix(h_bow_feature, 'BOW'))
        h_feature.update(add_feature_prefix(h_emb_feature, 'Emb'))
        return h_feature


    def _filter_e_grid(self, e_id, l_e_grid):
        """
        filer e grid to those that
            contain e id
            not too long (<self.max_sent_len)
        :param e_id: target e id
        :param l_e_grid: grid of doc
        :return:
        """
        l_kept_grid = []
        for e_grid in l_e_grid:
            if len(e_grid['sent'].split()) > self.max_sent_len:
                continue
            contain_flag = False
            for ana in e_grid[SPOT_FIELD]:
                if ana['id'] == e_id:
                    contain_flag = True
                    break
            if contain_flag:
                l_kept_grid.append(e_grid)
        return l_kept_grid

    def _form_nlss_bow(self, l_qe_nlss):
        l_sent = [nlss[0] for nlss in l_qe_nlss]
        return self._form_sents_bow(l_sent)

    def _form_nlss_emb(self, l_qe_nlss):
        l_sent = [nlss[0] for nlss in l_qe_nlss]
        return self._form_sents_emb(l_sent)

    def _form_grid_bow(self, l_e_grid):
        l_sent = [grid['sent'] for grid in l_e_grid]
        return self._form_sents_bow(l_sent)

    def _form_grid_emb(self, l_e_grid):
        l_sent = [grid['sent'] for grid in l_e_grid]
        return self._form_sents_emb(l_sent)

    def _form_sents_emb(self, l_sent):
        l_emb = [avg_embedding(self.resource.embedding, sent)
                 for sent in l_sent]
        return l_emb

    def _form_sents_bow(self, l_sent):
        l_h_lm = [text2lm(sent, clean=True) for sent in l_sent]
        return l_h_lm

    def _calc_bow_trans(self, l_bow_a, l_bow_b):
        # TODO
        m_trans = np.zeros((len(l_bow_a), l_bow_b))
        return m_trans

    def _calc_emb_trans(self, l_emb_a, l_emb_b):
        # TODO
        m_trans = np.zeros((len(l_emb_a), l_emb_b))
        return m_trans

    def _pool_grid_nlss_sim(self, trans_mtx):
        h_feature = {}
        # TODO max * sum in row * col, to one score
        return h_feature