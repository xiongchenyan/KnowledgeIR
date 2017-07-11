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
    add_feature_prefix,
    lm_cosine,
    TARGET_TEXT_FIELDS,
    body_field,
)
from knowledge4ir.utils.boe import (
    form_boe_per_field,
)
from knowledge4ir.utils.nlp import text2lm
from knowledge4ir.utils.retrieval_model import RetrievalModel


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
        self.intermediate_out = None
        if self.intermediate_data_out_name:
            self.intermediate_out = open(self.intermediate_data_out_name, 'w')

    def set_resource(self, resource):
        self.resource = resource
        assert resource.embedding
        assert resource.l_h_nlss
        while len(self.l_nlss_selection) < len(resource.l_h_nlss):
            self.l_nlss_selection.append('""')
        logging.info('%s feature resource set', self.feature_name_pre)

    def close_resource(self):
        if self.intermediate_data_out_name:
            self.intermediate_out.close()

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

        for p in xrange(len(ll_qe_nlss)):
            data = ll_qe_nlss[p]
            if type(data) is str:
                data = json.loads(data)
            nlss_name, l_qe_nlss, nlss_select = self.resource.l_nlss_name[p], data, self.l_nlss_selection[p]
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
            l_this_nlss = self._boe_nlss_filter(q_info, ana, l_nlss, doc_info)
        elif nlss_select == 'lm':
            l_this_nlss = self._lm_nlss_filter(l_nlss, doc_info)
        return l_this_nlss[:self.nb_nlss_per_e]

    def _boe_nlss_filter(self, q_info, q_ana, l_nlss, doc_info):
        e_id = q_ana['id']
        logging.info('filter [%d] nlss via boe', len(l_nlss))
        l_ana = sum([form_boe_per_field(doc_info, field) for field in self.l_target_fields],
                    [])
        s_e = set([ana['id'] for ana in l_ana if ana['id'] != e_id])
        h_e_sf = dict([(ana['id'], ana['surface']) for ana in l_ana])
        l_keep_nlss = []
        l_keep_nlss_e = []
        for nlss in l_nlss:
            keep_flag = False
            meet_e = None
            for e in nlss[1]:
                if e in s_e:
                    keep_flag = True
                    meet_e = e
                    break
            if keep_flag:
                l_keep_nlss.append(nlss)
                l_keep_nlss_e.append({'matched_e': [meet_e, h_e_sf[meet_e]]})
        if self.intermediate_out:
            h = {}
            h['qid'] = q_info['qid']
            h['ana'] = q_ana
            h['docno'] = doc_info['docno']
            h['boe_nlss'] = zip(l_keep_nlss_e, l_keep_nlss)
            print >> self.intermediate_out, json.dumps(h)
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
        if l_nlss_lmscore:
            logging.info('best lm [%f]', l_nlss_lmscore[0][1])
        return l_this_nlss

    def _form_nlss_bow(self, l_qe_nlss):
        l_sent = [nlss[0] for nlss in l_qe_nlss]
        return self._form_sents_bow(l_sent)

    def _form_nlss_emb(self, l_qe_nlss):
        l_sent = [nlss[0] for nlss in l_qe_nlss]
        return self._form_sents_emb(l_sent)

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
        logging.WARN('needs implement this function in inherited class')
        # raise NotImplementedError

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

    def _extract_retrieval_scores(self, formed_q_lm, formed_doc_lm, field):
        r_model = RetrievalModel()
        r_model.set_from_raw(
            formed_q_lm, formed_doc_lm,
            self.resource.corpus_stat.h_field_df.get(field, None),
            self.resource.corpus_stat.h_field_total_df.get(field, None),
            self.resource.corpus_stat.h_field_avg_len.get(field, None)
        )
        return [(k, v) for k, v in r_model.scores() if 'lm_twoway' != k]

    def _extract_simple_scores(self, formed_q_lm, formed_doc_lm):
        r_model = RetrievalModel()
        r_model.set_from_raw(
            formed_q_lm, formed_doc_lm,
        )
        l_score = [['cosine', lm_cosine(formed_q_lm, formed_doc_lm)],
                   ['coordinate', r_model.coordinate()]]
        return l_score