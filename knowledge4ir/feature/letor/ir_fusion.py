"""
classic features
"""
import math

from traitlets import List, Unicode

from knowledge4ir.feature import (
    calc_term_stat,
    LeToRFeatureExtractor,
    fetch_corpus_stat,
    fetch_doc_lm,
    TermStat
)
import logging
import json
from knowledge4ir.utils.nlp import text2lm


class LeToRIRFusionFeatureExtractor(LeToRFeatureExtractor):
    """
    extract the IR fusion features
    """
    feature_name_pre = Unicode('IRFusion')
    l_text_fields = List(Unicode, default_value=[]).tag(config=True)
    l_model = List(Unicode,
                   default_value=['lm_dir', 'bm25', 'coordinate', 'tf_idf']
                   ).tag(config=True)
    
    def __init__(self, **kwargs):
        super(LeToRIRFusionFeatureExtractor, self).__init__(**kwargs)
        self.s_model = set(self.l_model)

    def extract_for_text(self, query, docno, h_q_info, h_doc_info):
        h_feature = {}
        # logging.info('extracting IR fusion for q [%s], doc [%s]', query, docno)
        # logging.info('q_info %s', json.dumps(h_q_info))
        # logging.info('doc_info %s', json.dumps(h_doc_info))
        if 'term_vectors' not in h_doc_info:
            logging.warn('doc [%s] has no term vector', docno)
        if query == h_q_info['query']:
            h_old_feature = {}
            # title_old_ts = None
            for target_field in self.l_text_fields:
                term_stat = calc_term_stat(h_q_info, h_doc_info, target_field)
                # if target_field == 'title':
                #     title_old_ts = term_stat
                l_name_score = term_stat.mul_scores()
                for name, score in l_name_score:
                    if name in self.s_model:
                        feature_name = self.feature_name_pre + name.title() + target_field.title()
                        h_old_feature[feature_name] = score
            return h_old_feature

        h_tf = text2lm(query.lower())
        # title_ts = None
        for field in self.l_text_fields:
            total_df, avg_doc_len = fetch_corpus_stat(h_q_info, field)
            h_doc_tf, h_doc_df = fetch_doc_lm(h_doc_info, field)
            term_stat = TermStat()
            term_stat.set_from_raw(h_tf, h_doc_tf, h_doc_df, total_df, avg_doc_len)
            # if field == 'title':
            #     title_ts = term_stat
            l_sim_score = term_stat.mul_scores()
            for sim, score in l_sim_score:
                if sim in self.s_model:
                    feature_name = self.feature_name_pre + sim.title() + field.title()
                    h_feature[feature_name] = score
        #
        # for feature, score in h_feature.items():
        #     if score != h_old_feature[feature]:
        #         logging.warn('ltr feature value different')
        #         logging.warn('old feature: %s', json.dumps(h_old_feature))
        #         logging.warn('new feature: %s', json.dumps(h_feature))
        #
        #         logging.warn('old ts: %s', title_old_ts.pretty_print())
        #         logging.warn('new ts: %s', title_ts.pretty_print())
        #         logging.warn('query: %s, h_tf: %s', query, json.dumps(h_tf))
        #         break

        return h_feature

    def extract(self, qid, docno, h_q_info, h_doc_info):
        query = h_q_info['query']
        return self.extract_for_text(query, docno, h_q_info, h_doc_info)

