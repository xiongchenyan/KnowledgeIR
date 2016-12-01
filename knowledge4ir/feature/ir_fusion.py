"""
classic features
"""
from traitlets import List, Unicode

from knowledge4ir.feature import (
    LeToRFeatureExtractor,
    TermStat
)
import logging
import json
from knowledge4ir.utils.nlp import text2lm
from knowledge4ir.utils import load_corpus_stat
from knowledge4ir.utils import TARGET_TEXT_FIELDS


class LeToRIRFusionFeatureExtractor(LeToRFeatureExtractor):
    """
    extract the IR fusion features
    """
    feature_name_pre = Unicode('IRFusion')
    l_text_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_model = List(Unicode,
                   default_value=['lm_dir', 'bm25', 'coordinate', 'tf_idf']
                   ).tag(config=True)
    corpus_stat_pre = Unicode(help="the file pre of corpus stats").tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRIRFusionFeatureExtractor, self).__init__(**kwargs)
        self.s_model = set(self.l_model)
        self.h_field_h_df = dict()
        if self.corpus_stat_pre:
            l_field_h_df, self.h_corpus_stat = load_corpus_stat(
                self.corpus_stat_pre, self.l_text_fields)
            self.h_field_h_df = dict(l_field_h_df)
        for field in self.l_text_fields:
            assert field in self.h_corpus_stat
            assert field in self.h_field_h_df

    def set_external_info(self, external_info):
        super(LeToRIRFusionFeatureExtractor, self).set_external_info(external_info)
        self.h_field_h_df = external_info.h_field_h_df
        self.h_corpus_stat = external_info.h_corpus_stat
        return

    def extract_for_text(self, query, docno, h_q_info, h_doc_info):
        h_feature = {}
        # logging.info('extracting IR fusion for q [%s], doc [%s]', query, docno)
        # logging.info('q_info %s', json.dumps(h_q_info))
        # logging.info('doc_info %s', json.dumps(h_doc_info))

        h_tf = text2lm(query.lower())
        # title_ts = None
        for field in self.l_text_fields:
            total_df = self.h_corpus_stat[field]['total_df']
            avg_doc_len = self.h_corpus_stat[field]['average_len']
            h_doc_df = self.h_field_h_df[field]
            h_doc_tf = {}
            if field in h_doc_info:
                h_doc_tf = text2lm(h_doc_info[field].lower())

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

    def extract_doc_feature(self,docno, h_doc_info):
        h_feature = {}
        if 'is_wiki' in self.s_model:
            score = 0
            if 'enwp' in docno:
                score = 1
            h_feature[self.feature_name_pre + 'IsWiki'] = score
        return h_feature

    def extract(self, qid, docno, h_q_info, h_doc_info):
        query = h_q_info['query']
        h_feature = self.extract_for_text(query, docno, h_q_info, h_doc_info)
        h_feature.update(self.extract_doc_feature(docno, h_q_info))
        return h_feature



