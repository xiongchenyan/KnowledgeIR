"""
features from query itself, to document's bag-of-entities
    fill the 4 way matrix

a subclass of LeToRFeatureExtractor

features:
    q-doc e textual similarities
        bins, and top k


"""


from knowledge4ir.feature import (
    LeToRFeatureExtractor,
    TermStat,
    load_entity_texts,
)
from traitlets import (
    Unicode,
    List,
    Int
)
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_corpus_stat,
    text2lm,
    term2lm,
)
from knowledge4ir.utils import TARGET_TEXT_FIELDS
import logging
import json


class LeToRQDocETextFeatureExtractorC(LeToRFeatureExtractor):
    feature_name_pre = Unicode('ERank')
    l_text_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_model = List(Unicode,
                   default_value=['lm_dir', 'coordinate', 'tf_idf']
                   ).tag(config=True)
    l_pooling = List(Unicode,
                     default_value=['topk']).tag(config=True)
    top_k = Int(5, help='top 5 most similar entities to count').tag(config=True)

    l_entity_fields = List(Unicode, default_value=['desp'])
    entity_text_in = Unicode(help="entity texts in").tag(config=True)
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)
    corpus_stat_pre = Unicode(help="the file pre of corpus stats").tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRQDocETextFeatureExtractorC, self).__init__(**kwargs)
        self.h_corpus_stat = {}
        self.h_field_df = {}
        self._load_corpus_stat()
        self.h_entity_texts = {}
        if self.entity_text_in:
            self.h_entity_texts = load_entity_texts(self.entity_text_in)
        self.s_model = set(self.l_model)

    def _load_corpus_stat(self):
        if not self.corpus_stat_pre:
            logging.info('no corpus stat to load')
            return
        l_field_h_df, self.h_corpus_stat = load_corpus_stat(
            self.corpus_stat_pre, self.l_text_fields)
        self.h_field_h_df = dict(l_field_h_df)
        for field in self.l_text_fields:
            assert field in self.h_corpus_stat
            assert field in self.h_field_h_df

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = {}
        query = h_q_info['query']
        l_h_doc_e_lm = self._form_doc_e_lm(h_doc_info)
        l_e = sum([h.keys() for h in l_h_doc_e_lm], [])
        h_doc_e_texts = self._prepare_doc_e_texts(l_e)

        h_feature.update(self._extract_q_doc_e_textual_features(query, l_h_doc_e_lm, h_doc_e_texts))
        return h_feature

    def _form_doc_e_lm(self, h_doc_info):
        l_h_doc_e_lm = []
        for field in self.l_text_fields:
            l_e = []
            if field in h_doc_info[self.tagger]:
                l_e = [ana[0] for ana in h_doc_info[self.tagger][field]]
            h_lm = term2lm(l_e)
            l_h_doc_e_lm.append(h_lm)
        return l_h_doc_e_lm

    def _prepare_doc_e_texts(self, l_e):
        h_doc_e_texts = {}
        for e in l_e:
            h_fields = self.h_entity_texts.get(e, {})
            for key in h_fields.keys():
                if type(h_fields[key]) == list:
                    h_fields[key] = ' '.join(h_fields[key])
            h_doc_e_texts[e] = h_fields
        return h_doc_e_texts

    def _extract_q_doc_e_textual_features(self, query, l_h_doc_e_lm, h_doc_e_texts):
        if not self.h_entity_texts:
            return {}
        h_feature = {}
        q_lm = text2lm(query)
        for field, h_doc_e_lm in zip(self.l_text_fields, l_h_doc_e_lm):
            total_df = self.h_corpus_stat[field]['total_df']
            avg_doc_len = self.h_corpus_stat[field]['average_len']
            h_doc_df = self.h_field_h_df[field]
            l_h_scores = []
            l_e_tf = []
            for e, e_tf in h_doc_e_lm.items():
                h_scores = {}
                l_e_tf.append(e_tf)
                h_e_texts = h_doc_e_texts.get(e, {})
                for e_field in self.l_entity_fields:
                    text = h_e_texts.get(e_field, "")
                    e_lm = text2lm(text, clean=True)
                    term_stat = TermStat()
                    term_stat.set_from_raw(q_lm, e_lm, h_doc_df, total_df, avg_doc_len)
                    l_sim_score = term_stat.mul_scores()
                    for sim, score in l_sim_score:
                        if sim in self.l_model:
                            h_scores[e_field.title() + sim.title()] = score

                l_h_scores.append(h_scores)

            h_pooled_scores = self._merge_entity_sim(l_h_scores, l_e_tf)

            for name, score in h_pooled_scores.items():
                h_feature[self.feature_name_pre + field.title() + name] = score
        # logging.debug(json.dumps(h_feature))
        return h_feature

    def _merge_entity_sim(self, l_h_scores, l_e_tf):
        """
        merge scores in l_h_scores, with weights in l_e_tf
        :param l_h_scores:
        :param l_e_tf:
        :return:
        """
        h_pooled_scores = {}
        if 'max' in self.l_pooling:
            h_pooled_scores.update(self._max_pool_entity_sim(l_h_scores))
        if 'tf' in self.l_pooling:
            h_pooled_scores.update(self._wsum_pool_entity_sim(l_h_scores, l_e_tf))
        if 'topk' in self.l_pooling:
            h_pooled_scores.update(self._topk_pool_entity_sim(l_h_scores))
        return h_pooled_scores

    @classmethod
    def _max_pool_entity_sim(cls, l_h_scores):
        h_max = {}
        for h_scores in l_h_scores:
            for key, score in h_scores.items():
                h_max['Max' + key] = max(score, h_max.get(key, None))
        return h_max

    @classmethod
    def _wsum_pool_entity_sim(cls, l_h_scores, l_e_tf):
        h_wsum = {}
        z = sum(l_e_tf)
        for h_scores, w in zip(l_h_scores, l_e_tf):
            if z != 0:
                w /= float(z)
            for key, score in h_scores.items():
                h_wsum['Wsum' + key] = score * w + h_wsum.get(key, 0)
        return h_wsum

    def _topk_pool_entity_sim(self, l_h_scores):
        h_topk = {}
        h_key_scores = {}
        for h_scores in l_h_scores:
            for key, score in h_scores.items():
                if key not in h_key_scores:
                    h_key_scores[key] = [score]
                else:
                    h_key_scores[key].append(score)

        for key, l_score in h_key_scores.items():
            l_score.sort(reverse=True)
            while len(l_score) < self.top_k:
                l_score.append(-20)
            for k in xrange(self.top_k):
                h_topk[key + 'Top%d' % (k + 1)] = l_score[k]
        return h_topk

