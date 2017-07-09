"""
I am the basic functions used for retrieval models

the base class for retrieval models
bm25
cosine
lm
"""

import json
import logging
import numpy as np
from traitlets import (
    Int, Float, List, Unicode, Set
)
import pickle
import sys
from traitlets.config import Configurable
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


BM25_K1 = 1.2
BM25_B = 0.75
BM25_K3 = 7
LM_DIR_MU = 2500
LM_MIN_TF = 0.1
LM_JM_LAMBDA = 0.4
MIN_LM_SCORE = 1e-10


class BM25Para(Configurable):
    """
    bm25's para
    """
    k1 = Float(BM25_K1).tag(config=True)
    b = Float(BM25_B).tag(config=True)


class LmPara(Configurable):
    dir_mu = Int(LM_DIR_MU).tag(config=True)
    min_tf = Float(LM_MIN_TF).tag(config=True)
    jm_lambda = Float(LM_JM_LAMBDA).tag(config=True)


class CorpusStat(Configurable):
    """
    the corpus stat
    field -> h_df
    field -> total_df
    field -> avg_doc_len
    """
    corpus_stat_in = Unicode(help="location of prepared corpus stat").tag(config=True)

    def __init__(self, **kwargs):
        super(CorpusStat, self).__init__(**kwargs)
        if self.corpus_stat_in:
            logging.info('loading corpus stat in [%s]...', self.corpus_stat_in)
            l_data = pickle.load(open(self.corpus_stat_in, 'rb'))
            self.h_field_df, self.h_field_total_df, self.h_field_avg_len = l_data
            logging.info('corpus stat in [%s] loaded', self.corpus_stat_in)


class RetrievalModel(Configurable):
    """
    the retrieval models for a query-doc field pair
    """
    default_model_group = Set(Unicode,
                              default_value={'lm_dir', 'lm_twoway',
                                             'bm25', 'coordinate', 'tf_idf'}
                              ).tag(config=True)

    def __init__(self, **kwargs):
        super(RetrievalModel, self).__init__(**kwargs)
        self.v_q_df = np.zeros(0)
        self.v_tf = np.zeros(0)
        self.doc_len = 0.0
        self.v_q_tf = np.zeros(0)
        self.total_df = 0.0
        self.avg_doc_len = 0.0

    def set_dim(self, d):
        self.v_q_df = np.zeros(d)
        self.v_tf = np.zeros(d)
        self.v_q_tf = np.zeros(d)

    def set(self, h_q_terms, h_doc_terms, field, corpus_stat):
        """
        set model using corpus stat and field
        :param h_q_terms: q tf lm
        :param h_doc_terms: d tf lm
        :param field: field to load from corpus_stat
        :param corpus_stat: CorpusStat
        :return:
        """
        assert field in corpus_stat.h_field_df
        assert field in corpus_stat.h_field_total_df
        assert field in corpus_stat.h_field_avg_len
        h_df  = corpus_stat.h_field_df[field]
        total_df = corpus_stat.h_field_total_df[field]
        avg_doc_len = corpus_stat.h_field_avg_len[field]
        return self.set_from_raw(h_q_terms, h_doc_terms, h_df, total_df, avg_doc_len)

    def set_from_raw(self, h_q_terms, h_doc_terms, h_df={}, total_df=None, avg_doc_len=None):
        """
        set term stat by raw data
        :param h_q_terms: query term -> tf
        :param h_doc_terms: doc term -> tf
        :param h_df: term ->df dict
        :param total_df: a int of total document frequency
        :param avg_doc_len: a float of avg document length
        :return:
        """
        l_q_terms_tf = h_q_terms.items()
        l_q_terms = [item[0] for item in l_q_terms_tf]
        self.set_dim(len(l_q_terms))
        l_q_tf = [float(item[1]) for item in l_q_terms_tf]
        self.doc_len = float(sum([item[1] for item in h_doc_terms.items()]))
        if total_df:
            self.total_df = float(total_df)
        if avg_doc_len:
            self.avg_doc_len = float(avg_doc_len)
        l_q_df = []
        l_doc_tf = []

        for q_t in l_q_terms:
            if q_t in h_df:
                l_q_df.append(float(h_df[q_t]))
            else:
                # logging.debug('%s no df', q_t)
                l_q_df.append(0.0)
            if q_t in h_doc_terms:
                l_doc_tf.append(float(h_doc_terms[q_t]))
            else:
                l_doc_tf.append(0)

        self.v_q_tf = np.array(l_q_tf)
        self.v_q_df = np.array(l_q_df)
        self.v_tf = np.array(l_doc_tf)

        return

    def pretty_print(self):
        resp = "q_tf: %s\n" % np.array2string(self.v_q_tf, precision=4)
        resp += 'q_df: %s\n' % np.array2string(self.v_q_df, precision=4)
        resp += 'tf: %s\n' % np.array2string(self.v_tf, precision=4)
        resp += 'doc_len: %d\n' % self.doc_len
        resp += 'total_df: %d\n' % self.total_df
        resp += 'avg_doc_len: %.2f' % self.avg_doc_len
        return resp

    def scores(self):
        l_name_score = self.all_scores()
        l_score = [(name, score)
                   for name, score in l_name_score if name in self.default_model_group]
        return l_score

    def all_scores(self, lm_para=LmPara(), bm25_para=BM25Para()):
        """
        the entrance of all my similarity functions
        :param lm_para:
        :param bm25_para:
        :return:
        """
        # l_sim_func = ['lm', 'lm_dir', 'lm_jm', 'lm_twoway',
        #               'bm25', 'coordinate', 'cosine', 'tf_idf',
        #               'bool_and', 'bool_all']
        l_name_score = list()
        l_name_score.append(['lm', self.lm(lm_para)])
        l_name_score.append(['lm_dir', self.lm_dir(lm_para)])
        l_name_score.append(['lm_jm', self.lm_jm(lm_para)])
        l_name_score.append(['lm_twoway', self.lm_twoway(lm_para)])

        l_name_score.append(['bm25', self.bm25(bm25_para)])

        l_name_score.append(['coordinate', self.coordinate()])
        l_name_score.append(['tf_idf', self.tf_idf()])
        l_name_score.append(['bool_and', self.bool_and()])
        l_name_score.append(['bool_or', self.bool_or()])

        return l_name_score

    def simple_scores(self):
        l_name_score = list()
        l_name_score.append(['lm', self.lm()])
        l_name_score.append(['coordinate', self.coordinate()])
        l_name_score.append(['bool_and', self.bool_and()])
        l_name_score.append(['bool_or', self.bool_or()])

        return l_name_score

    def lm(self, lm_para=LmPara()):
        """
        return lm score for myself
        :return:
        """
        if self.doc_len == 0:
            return np.log(MIN_LM_SCORE)
        v_tf = np.maximum(self.v_tf, lm_para.min_tf)
        v_tf /= self.doc_len
        v_tf = np.maximum(v_tf, MIN_LM_SCORE)
        score = np.log(v_tf).dot(self.v_q_tf)

        return score

    def lm_dir(self, lm_para=LmPara()):
        if self.doc_len == 0:
            return np.log(MIN_LM_SCORE)
        v_q = self.v_q_tf / float(np.sum(self.v_q_tf))
        v_mid = (self.v_tf + lm_para.dir_mu * (self.v_q_df / self.total_df)) / (self.doc_len +
                                                                                lm_para.dir_mu)
        v_mid = np.maximum(v_mid, MIN_LM_SCORE)
        score = np.log(v_mid).dot(v_q)
        return score + 20

    def lm_jm(self, lm_para=LmPara()):
        if self.doc_len == 0:
            return np.log(MIN_LM_SCORE)

        v_mid = self.v_tf / self.doc_len * (1 - lm_para.jm_lambda) \
                + lm_para.jm_lambda * self.v_q_df / self.total_df
        v_mid = np.maximum(v_mid, MIN_LM_SCORE)
        score = np.log(v_mid).dot(self.v_q_tf)
        return score + 20

    def lm_twoway(self, lm_para=LmPara()):
        if self.doc_len == 0:
            return np.log(MIN_LM_SCORE)

        v_mid = (self.v_tf + lm_para.dir_mu * (self.v_q_df / self.total_df)) \
                / (self.doc_len + lm_para.dir_mu)
        v_mid = v_mid * (1 - lm_para.jm_lambda) \
                + lm_para.jm_lambda * self.v_q_df / self.total_df
        v_mid = np.maximum(v_mid, MIN_LM_SCORE)
        score = np.log(v_mid).dot(self.v_q_tf)
        return score + 20

    def bm25(self, bm25_para=BM25Para()):
        if self.doc_len == 0:
            return 0
        v_q = self.v_q_tf / float(np.sum(self.v_q_tf))
        v_tf_part = self.v_tf * (bm25_para.k1 + 1) \
                    / (self.v_tf + bm25_para.k1 * (1 - bm25_para.b + bm25_para.b * self.doc_len /
                                                   self.avg_doc_len))

        v_mid = (self.total_df - self.v_q_df + 0.5) / (self.v_q_df + 0.5)
        v_mid = np.maximum(v_mid, 1.0)
        v_idf_q = np.log(v_mid)
        v_idf_q = np.maximum(v_idf_q, 0)
        score = v_mid.dot(v_tf_part * v_idf_q)
        score = max(score, 1.0)
        score = np.log(score)
        return score

    def coordinate(self):
        return sum(self.v_tf > 0)

    def bool_and(self):
        if self.coordinate() == len(self.v_q_tf):
            return 1
        return 0

    def bool_or(self):
        return min(1, self.coordinate())

    def tf_idf(self):
        if self.doc_len == 0:
            return 0

        normed_idf = np.log(1 + self.total_df / np.maximum(self.v_q_df, 1))
        normed_tf = self.v_tf / self.doc_len
        return normed_idf.dot(normed_tf)

    def tf(self):
        if self.doc_len == 0:
            return 0
        normed_tf = self.v_tf / self.doc_len
        return sum(normed_tf)
