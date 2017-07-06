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
    body_field,
    E_GRID_FIELD,
    add_feature_prefix,
    text2lm,
    avg_embedding,
    SPOT_FIELD,
    lm_cosine,
    TARGET_TEXT_FIELDS,
    max_pool_feature,
)
from knowledge4ir.utils.retrieval_model import RetrievalModel


class NLSSFeature(BoeFeature):
    """
    root class for nlss features
    """
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
        logging.info('Salient feature resource set')

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
            h_this_nlss_feature = self._extract_per_entity_via_nlss(q_info, ana, doc_info, l_qe_nlss)
            h_feature.update(add_feature_prefix(h_this_nlss_feature, nlss_name + '_'))
        return h_feature

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


class EGridNLSSFeature(NLSSFeature):
    """
    extract boe exact features by comparing e_grid of qe with qe's nlss
    """
    feature_name_pre = Unicode('EGridNLSS')

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
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
            l_e_grid = h_e_grid.get(field, [])
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

        self._log_intermediate_res(ana, doc_info, l_this_e_grid, l_qe_nlss, m_bow_sim, m_emb_sim)

        h_bow_feature = self._pool_grid_nlss_sim(m_bow_sim)
        h_emb_feature = self._pool_grid_nlss_sim(m_emb_sim)

        h_feature = dict()
        h_feature.update(add_feature_prefix(h_bow_feature, 'BOW_'))
        h_feature.update(add_feature_prefix(h_emb_feature, 'Emb_'))
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

    def _form_grid_bow(self, l_e_grid):
        l_sent = [grid['sent'] for grid in l_e_grid]
        return self._form_sents_bow(l_sent)

    def _form_grid_emb(self, l_e_grid):
        l_sent = [grid['sent'] for grid in l_e_grid]
        return self._form_sents_emb(l_sent)

    def _pool_grid_nlss_sim(self, trans_mtx):
        h_feature = {}
        l_func = [np.mean, np.amax]
        l_name = ['Mean', 'Max']
        for f1, name1 in zip(l_func + [np.sum], l_name + ['Sum']):
            for f2, name2 in zip(l_func, l_name):
                score = -1
                if (trans_mtx.shape[0] > 0) & (trans_mtx.shape[1] > 0):
                    score = f1(f2(trans_mtx, axis=1), axis=0)
                pool_name = 'R' + name1 + 'C' + name2
                h_feature[pool_name] = score


        return h_feature

    def _log_intermediate_res(self, ana, doc_info, l_this_e_grid, l_qe_nlss, m_bow_sim, m_emb_sim):
        """
        dump out the intermediate results
            e_id, name, doc no,
            e_grid_sentences:
            grid sentence for this e_id in doc, mean sim in bow and emb,
                max sim in bow and emb, and corresponding nlss that generate the max
        :param ana:
        :param doc_info:
        :param l_this_e_grid:
        :param l_qe_nlss:
        :param m_bow_sim:
        :param m_emb_sim:
        :return:
        """
        # use json
        if not doc_info:
            return
        h_pair_res = dict()
        h_pair_res['id'] = ana['id']
        h_pair_res['surface'] = ana['surface']
        h_pair_res['docno'] = doc_info['docno']
        if (not l_this_e_grid) | (not l_qe_nlss):
            print >> self.intermediate_out, json.dumps(h_pair_res)
            return

        l_e_grid_info = []
        for i in xrange(len(l_this_e_grid)):
            h_this_sent = {}
            h_this_sent['sent'] = l_this_e_grid[i]['sent']
            h_this_sent['mean_bow_sim'] = np.mean(m_bow_sim[i])
            h_this_sent['mean_emb_sim'] = np.mean(m_emb_sim[i])

            max_p = np.argmax(m_bow_sim[i])
            h_this_sent['max_bow_sim'] = m_bow_sim[i, max_p]
            h_this_sent['max_bow_nlss'] = l_qe_nlss[max_p][0]
            max_p = np.argmax(m_emb_sim[i])
            h_this_sent['max_emb_sim'] = m_emb_sim[i, max_p]
            h_this_sent['max_emb_nlss'] = l_qe_nlss[max_p][0]

            l_e_grid_info.append(h_this_sent)

        h_pair_res['e_grid_info'] = l_e_grid_info

        print >> self.intermediate_out, json.dumps(h_pair_res)
        return


class NLSSExpansionFeature(NLSSFeature):
    """
    find best nlss for the query (using embedding cosine)
        top 5 for now
    and then use them to rank the document via qe-dw
        top 5 nlss combined as e-desp-alike big query
        each nlss individually, and then take a max
    also dump the top k nlss used
    """
    top_k_nlss = Int(5, help='number of nlss to use per query entity').tag(config=True)
    feature_name_pre = Unicode('NLSSExp')

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
        """
        extract e-d features

        do:
            get top k nlss
            form doc lm
            retrieval, as a whole of individually
            sum up to features
        :param q_info: query info
        :param ana:
        :param doc_info:
        :param l_qe_nlss:
        :return: h_feature: entity features for this nlss set
        """

        l_top_nlss = self._find_top_k_nlss_for_q(q_info, ana, l_qe_nlss)

        l_top_sent = [nlss[0] for nlss in l_top_nlss]
        l_top_sent.append(' '.join(l_top_sent))
        if not l_top_sent:
            l_top_sent.append('')  # place holder for empty nlss e
        l_h_per_sent_feature = []
        l_field_doc_lm = [text2lm(doc_info.get(field, ""), clean=True)
                          for field in self.l_target_fields]
        for sent in l_top_sent:
            h_per_sent_feature = {}
            h_sent_lm = text2lm(sent, clean=True)
            for field, lm in zip(self.l_target_fields, l_field_doc_lm):
                r_model = RetrievalModel()
                r_model.set_from_raw(
                    h_sent_lm, lm,
                    self.resource.corpus_stat.h_field_df.get(field, None),
                    self.resource.corpus_stat.h_field_total_df.get(field, None),
                    self.resource.corpus_stat.h_field_avg_len.get(field, None)
                )
                l_retrieval_score = r_model.scores()
                h_per_sent_feature.update(dict(
                    [(field + name, score) for name, score in l_retrieval_score]
                ))
            l_h_per_sent_feature.append(h_per_sent_feature)

        h_max_feature = max_pool_feature(l_h_per_sent_feature[:-1])
        h_mean_feature = add_feature_prefix(l_h_per_sent_feature[-1], 'Conca')

        h_feature = h_max_feature
        h_feature.update(h_mean_feature)
        return h_feature

    def _find_top_k_nlss_for_q(self, q_info, ana, l_qe_nlss):
        """
        find top k similar sentences based on cosine(q emb, sent emb)
        :param q_info: query info
        :param ana: current q e
        :param l_qe_nlss: nlss of this e
        :return:
        """

        query = q_info[QUERY_FIELD]
        q_emb = avg_embedding(self.resource.embedding, query)
        l_nlss_emb = self._form_nlss_emb(l_qe_nlss)

        m_emb_sim = self._calc_emb_trans([q_emb], l_nlss_emb)
        l_emb_sim_score = m_emb_sim[0].tolist()
        l_nlss_with_score = zip(l_qe_nlss, l_emb_sim_score)
        l_nlss_with_score.sort(key=lambda item: item[1], reverse=True)
        l_top_nlss = [item[0] for item in l_nlss_with_score[:self.top_k_nlss]]

        self._log_qe_top_nlss(q_info, ana, l_top_nlss)

        return l_top_nlss

    def _log_qe_top_nlss(self, q_info, ana, l_top_nlss):
        """
        dump a packed intermediate information in it
        """
        h_info = dict(q_info)
        h_info['current_e'] = ana
        h_info['top_nlss'] = l_top_nlss
        print >> self.intermediate_out, json.dumps(h_info)
