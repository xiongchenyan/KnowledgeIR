"""
les features
basically the textual fields of query entity to t
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
import json


class LeToRLesFeatureExtractor(LeToRFeatureExtractor):
    feature_name_pre = Unicode('Les')
    l_text_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_model = List(Unicode,
                   default_value=['lm_dir', 'bm25', 'coordinate', 'tf_idf']
                   ).tag(config=True)
    corpus_stat_pre = Unicode(help="the file pre of corpus stats").tag(config=True)
    l_entity_fields = List(Unicode, default_value=['name', 'alias', 'desp'])
    entity_text_in = Unicode(help="entity texts in").tag(config=True)
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRLesFeatureExtractor, self).__init__(**kwargs)
        self.h_entity_texts = self._load_entity_texts()
        self.s_model = set(self.l_model)
        l_field_h_df, self.h_corpus_stat = load_corpus_stat(
            self.corpus_stat_pre, self.l_text_fields)
        self.h_field_h_df = dict(l_field_h_df)
        for field in self.l_text_fields:
            assert field in self.h_corpus_stat
            assert field in self.h_field_h_df

    def _load_entity_texts(self):
        h = {}
        logging.info('loading entity texts from [%s]', self.entity_text_in)
        for line_cnt, line in enumerate(open(self.entity_text_in)):
            h_e = json.loads(line)
            h[h_e['id']] = h_e
            if not line_cnt % 1000:
                logging.info('loaded [%d] entities texts', line_cnt)
        logging.info('finished loading [%d] entities texts', len(h))
        return h

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = {}

        l_q_e = [ana[0] for ana in h_q_info[self.tagger]['query']]

        for field in self.l_text_fields:
            total_df = self.h_corpus_stat[field]['total_df']
            avg_doc_len = self.h_corpus_stat[field]['average_len']
            h_doc_df = self.h_field_h_df[field]
            h_doc_tf = {}
            if field in h_doc_info:
                h_doc_tf = text2lm(h_doc_info[field].lower())

            for e_field in self.l_entity_fields:
                h_sim_score = {}
                cnt = 0
                for e in l_q_e:
                    if e not in self.h_entity_texts:
                        continue
                    if e_field not in self.h_entity_texts[e]:
                        continue
                    e_text = self.h_entity_texts[e][e_field]
                    h_tf = text2lm(e_text, clean=True)
                    cnt += 1
                    term_stat = TermStat()
                    term_stat.set_from_raw(h_tf, h_doc_tf, h_doc_df, total_df, avg_doc_len)
                    # if field == 'title':
                    #     title_ts = term_stat
                    for sim, score in term_stat.mul_scores():
                        if sim in self.s_model:
                            if sim not in h_sim_score:
                                h_sim_score[sim] = score
                            else:
                                h_sim_score[sim] += score

                if cnt:
                    for sim in h_sim_score:
                        h_sim_score[sim] /= cnt
                for sim, score in h_sim_score.items():
                    h_feature[self.feature_name_pre + e_field.title() + field.title() + sim.title()] = score

        return h_feature



