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
)
from knowledge4ir.utils.boe import (
    form_boe_per_field,
)
from knowledge4ir.utils.retrieval_model import RetrievalModel
from knowledge4ir.utils.nlp import text2lm


class NLSSFeature(BoeFeature):
    """
    root class for nlss features
    """
    nb_nlss_per_e = Int(100, help='number of nlss per e').tag(config=True)
    l_nlss_selection = List(Unicode, default_value=[''],
                            help='way to select nlss for the q: ""(nothing), lm (rank via lm), BOE (has e overlap)'
                            ).tag(config=True)
    intermediate_data_out_name = Unicode(help='intermediate output results').tag(config=True)
    max_sent_len = Int(100, help='max grid sentence len to consider').tag(config=True)
    l_target_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)

    def __init__(self, **kwargs):
        super(NLSSFeature, self).__init__(**kwargs)
        if self.intermediate_data_out_name:
            self.intermediate_out = open(self.intermediate_data_out_name, 'w')

    def set_resource(self, resource):
        self.resource = resource
        assert resource.embedding
        assert resource.l_h_nlss
        logging.info('%s feature resource set', self.feature_name_pre)

    def close_resource(self):
        if self.intermediate_data_out_name:
            self.intermediate_out.close()

    def extract_pair(self, q_info, doc_info):
        """

        :param q_info:
        :param doc_info:
        :return:
        """
        logging.debug('extracting e_grid nlss features for [%s][%s]',
                      q_info['qid'], doc_info['docno'])
        l_q_ana = self._get_field_ana(q_info, QUERY_FIELD)
        logging.debug('q info %s', json.dumps(q_info))
        logging.debug('q ana %s', json.dumps(l_q_ana))
        logging.debug('doc t [%s], info [%s]', doc_info.get('title', ""),
                      json.dumps(doc_info.get('spot', {}).get('title', []))
                      )
        l_h_feature = [self.extract_per_entity(q_info, ana, doc_info) for ana in l_q_ana]

        h_final_feature = {}
        # h_final_feature.update(log_sum_feature(l_h_feature))
        h_final_feature.update(mean_pool_feature(l_h_feature))
        h_final_feature = dict([(self.feature_name_pre + item[0], item[1])
                                for item in h_final_feature.items()])

        return h_final_feature

    def extract_per_entity(self, q_info, ana, doc_info):
        """
        :param q_info: query info
        :param ana: one q ana
        :param doc_info:
        :return:
        """

        h_feature = dict()
        e_id = ana['id']
        ll_qe_nlss = [h_nlss.get(e_id, []) for h_nlss in self.resource.l_h_nlss]

        for nlss_name, l_qe_nlss in zip(self.resource.l_nlss_name, ll_qe_nlss):
            for nlss_select in self.l_nlss_selection:
                l_this_nlss = self._select_nlss(q_info, ana, doc_info, nlss_select, l_qe_nlss)
                h_this_nlss_feature = self._extract_per_entity_via_nlss(q_info, ana, doc_info, l_this_nlss)
                h_feature.update(add_feature_prefix(h_this_nlss_feature, nlss_select + nlss_name + '_'))
        return h_feature

    def _select_nlss(self, q_info, ana, doc_info, nlss_select, l_nlss):
        if nlss_select:
            logging.info('selection for q [%s] [%s] via [%s]',q_info['qid'], ana['id'], nlss_select)
        l_this_nlss = []
        if nlss_select == "":
            l_this_nlss = l_nlss
        elif nlss_select == "BOE":
            l_this_nlss = self._boe_nlss_filter(ana['id'], l_nlss, doc_info)
        elif nlss_select == 'lm':
            l_this_nlss = self._lm_nlss_filter(l_nlss, doc_info)
        return l_this_nlss[:self.nb_nlss_per_e]

    def _boe_nlss_filter(self, e_id, l_nlss, doc_info):
        logging.info('filter [%d] nlss via boe', len(l_nlss))
        l_ana = sum([form_boe_per_field(doc_info, field) for field in self.l_target_fields],
                    [])
        s_e = set([ana['id'] for ana in l_ana if ana['id'] != e_id])
        l_keep_nlss = []
        for nlss in l_nlss:
            keep_flag = False
            for e in nlss[1]:
                if e in s_e:
                    keep_flag = True
                    break
            if keep_flag:
                l_keep_nlss.append(nlss)
        logging.info('[%s] boe filtered [%d]->[%d]', e_id, len(l_nlss), len(l_keep_nlss))
        return l_keep_nlss

    def _lm_nlss_filter(self, l_nlss, doc_info):
        logging.info('filter [%d] nlss via boe', len(l_nlss))
        l_nlss_lmscore = []
        h_d_lm = text2lm(doc_info.get(body_field, ""))

        for nlss in l_nlss:
            h_s_lm = text2lm(nlss[0])
            r_model = RetrievalModel()
            r_model.set_from_raw(h_s_lm, h_d_lm)
            lm = r_model.lm()
            l_nlss_lmscore.append((nlss, lm))
        l_nlss_lmscore.sort(key=lambda item: item[1], reverse=True)
        l_this_nlss = [item[0] for item in l_nlss_lmscore]
        return l_nlss


    def _form_sents_emb(self, l_sent):
        l_emb = [avg_embedding(self.resource.embedding, sent)
                 for sent in l_sent]
        return l_emb

    def _form_sents_bow(self, l_sent):
        l_h_lm = [text2lm(sent, clean=True) for sent in l_sent]
        return l_h_lm

    def _form_nlss_bow(self, l_qe_nlss):
        l_sent = [nlss[0] for nlss in l_qe_nlss]
        return self._form_sents_bow(l_sent)

    def _form_nlss_emb(self, l_qe_nlss):
        l_sent = [nlss[0] for nlss in l_qe_nlss]
        return self._form_sents_emb(l_sent)

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
        raise NotImplementedError

    def _calc_bow_trans(self, l_bow_a, l_bow_b):
        m_trans = np.zeros((len(l_bow_a), len(l_bow_b)))
        for i in xrange(len(l_bow_a)):
            for j in xrange(len(l_bow_b)):
                m_trans[i, j] = lm_cosine(l_bow_a[i], l_bow_b[j])
        return m_trans

    def _calc_emb_trans(self, l_emb_a, l_emb_b):
        m_trans = np.zeros((len(l_emb_a), len(l_emb_b)))
        for i in xrange(len(l_emb_a)):
            if l_emb_a[i] is None:
                continue
            for j in xrange(len(l_emb_b)):
                if l_emb_b[j] is None:
                    continue
                m_trans[i, j] = 1 - cosine(l_emb_a[i], l_emb_b[j])
        return m_trans


