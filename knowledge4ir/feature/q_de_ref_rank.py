"""
features from query itself, to document's bag-of-entities
    fill the 4 way matrix

a subclass of LeToRFeatureExtractor

features:
    tf weighted q-doc e textual similarities
    coverage of doc e's name on query
    e's rank in query's reference entity ranking (indri, FACC1)


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


class LeToRQDocERefRankFeatureExtractorC(LeToRFeatureExtractor):
    feature_name_pre = Unicode('QDocERefRank')
    l_text_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)
    l_ref_rank = List(Unicode, help='query reference entity ranking').tag(config=True)
    l_ref_rank_name = List(Unicode, help='query reference rank name').tag(config=True)
    l_top_k = List(Int, default_value=[1, 5, 10, 20, 100],
                   help='ref rank overlap top k to consider'
                   ).tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRQDocERefRankFeatureExtractorC, self).__init__(**kwargs)
        self.h_corpus_stat = {}
        self.h_field_df = {}
        self.l_h_q_ref_ranking = [dict(load_trec_ranking_with_score(ranking_in))
                                  for ranking_in in self.l_ref_rank]

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = {}
        query = h_q_info['query']
        l_h_doc_e_lm = self._form_doc_e_lm(h_doc_info)
        l_e = sum([h.keys() for h in l_h_doc_e_lm], [])

        h_feature.update(self._extract_q_doc_e_ref_rank_feature(qid, l_h_doc_e_lm))

        return h_feature

    def _extract_q_doc_e_ref_rank_feature(self, qid, l_h_doc_e_lm):
        """
        check how many of the entities is in top 1, 10
        :param qid:
        :param l_h_doc_e_lm:
        :return:
        """
        h_feature = {}
        l_q_rank = [h_q_rank.get(qid, []) for h_q_rank in self.l_h_q_ref_ranking]
        l_q_ref_rank_p = []
        for rank in l_q_rank:
            h = dict(zip([doc for doc, __ in rank], range(1, len(l_q_rank) + 1)))
            l_q_ref_rank_p.append(h)

        for field, h_doc_e_lm in zip(self.l_text_fields, l_h_doc_e_lm):
            if field == 'bodyText':
                for ref_name, h_ref_rank_p in zip(self.l_ref_rank_name, l_q_ref_rank_p):
                    l_e_rank_p = []
                    for e in h_doc_e_lm.keys():
                        p = h_ref_rank_p.get(e, 10000000)
                        l_e_rank_p.append(p)
                    l_top_k_cnt = self._count_topk(l_e_rank_p, self.l_top_k)
                    for top_k, top_k_cnt in zip(self.l_top_k, l_top_k_cnt):
                        feature_name = self.feature_name_pre + ref_name.title()
                        h_feature[feature_name + 'Top%03d' % top_k] = top_k_cnt
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

    @classmethod
    def _count_topk(cls, l_ranks, l_top_k):
        l_top_k_cnt = []
        l_p = sorted(l_ranks)
        i = 0
        l_top_k_cnt.append(0)
        for p in l_p:
            if i >= len(l_top_k):
                break
            while p > l_top_k[i]:
                i += 1
                if i >= len(l_top_k):
                    break
                l_top_k_cnt.append(l_top_k_cnt[-1])

            l_top_k_cnt[-1] += 1
        while len(l_top_k_cnt) < len(l_top_k):
            l_top_k_cnt.append(l_top_k_cnt[-1])
        return l_top_k_cnt

