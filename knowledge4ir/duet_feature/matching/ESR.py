"""
embedding features for BOE
input:
    entity embedding in word2vec format
    query and doc's annotation
        optional: doc entity's attention

output:
    ESR features
"""

from gensim.models import Word2Vec
import numpy as np
import json
import logging
from knowledge4ir.duet_feature import LeToRFeatureExtractor
from traitlets import (
    Unicode,
    Int,
    List,
    Float,
    Bool,
)
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
)
import math
from knowledge4ir.utils import (
    bin_similarity,
    form_bins,
)


class ESRFeatureExtractor(LeToRFeatureExtractor):
    tagger = Unicode('spot', help='tagger used, as in q info and d info')
    l_target_fields = List(Unicode,
                           default_value=[TARGET_TEXT_FIELDS],
                           help='doc fields to use'
                           ).tag(config=True)
    embedding_in = Unicode(help='embedding data spot (word2vec format) if only one',
                           ).tag(config=True)
    distance = Unicode('cos', help='distance function, cos|l1'
                       ).tag(config=True)

    # top_k = Int(3,
    #             help="number of soft match feature per paper field"
    #             ).tag(config=True)
    feature_name_pre = Unicode('ESR')
    nb_bin = Int(6, help='number of similarity bins').tag(config=True)
    bin_func = Unicode('log',
                       help="the func to apply on the bin counts: log|tf|norm_tf"
                       ).tag(config=True)
    pool_func = List(Unicode, default_value=['max'],
                     help="pooling at query entities 'max', 'mean', 'mean-all', 'topk'"
                     ).tag(config=True)
    bin_range = Float(1.0, help="the bin range to keep in bin").tag(config=True)
    l_bins = List(Float, default_value=[],
                  help="bins to use, if given, will directly use this one"
                  ).tag(config=True)
    log_min = Float(1e-10, help='log of zero bin').tag(config=True)
    use_entity_weight = Bool(False, help='weight use the scores of doc entities').tag(config=True)

    def __init__(self, **kwargs):
        super(ESRFeatureExtractor, self).__init__(**kwargs)
        if self.embedding_in:
            logging.info('start loading embedding %s', json.dumps(self.embedding_in))
            self.embedding = Word2Vec.load_word2vec_format(self.embedding_in)
            logging.info('embedding loaded')
        if not self.l_bins:
            self.l_bins = form_bins(self.nb_bin, self.bin_range)
        logging.info('use bins %s', json.dumps(self.l_bins))

        self.h_pool_func = {
            'max': self._max_bin,
            'mean': self._mean_bin,
            # 'mean-all': self._mean_all,
            # 'topk': self._top_k_all
        }
        self.h_distance_func = {
            'cos': self._build_cosine_mtx,
            'l1': self._build_l1_mtx,
        }
        assert self.distance in self.h_distance_func

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = {}
        emb_model = self.embedding
        emb_name = ""
        l_q_e = [ana['entities'][0]['id'] for ana in h_q_info[self.tagger]['query']
                 if ana['entities'][0]['id'] in emb_model]
        for field, l_ana in h_doc_info[self.tagger].items():
            if field not in self.l_target_fields:
                continue
            l_doc_e = [ana['entities'][0]['id'] for ana in l_ana
                       if ana['entities'][0]['id'] in emb_model]
            l_doc_e_weight = []
            if self.use_entity_weight:
                l_doc_e_weight = [ana['entities'][0]['score']
                                  for ana in l_ana if ana['entities'][0]['id'] in emb_model]

            l_sim_mtx = []
            m_sim_mtx = self.h_distance_func[self.distance](l_q_e, l_doc_e, emb_model)
            l_sim_mtx.append(m_sim_mtx)

            l_total_bin_score = []
            for d in xrange(len(l_sim_mtx)):
                l_this_bin_score = []
                m_sim_mtx = l_sim_mtx[d]
                for pool_name in self.pool_func:
                    assert pool_name in self.h_pool_func
                    l_this_bin_score.extend(self.h_pool_func[pool_name](m_sim_mtx, l_doc_e_weight))
                if len(l_sim_mtx) > 1:
                    l_this_bin_score = [
                        ('D%03d' % d + item[0], item[1]) for item in l_this_bin_score]
                l_total_bin_score.extend(l_this_bin_score)

            for bin_name, score in l_total_bin_score:
                feature_name = '_'.join([self.feature_name_pre,
                                         emb_name,
                                         field.title(),
                                         bin_name.title()])
                h_feature[feature_name] = score

        return h_feature

    @classmethod
    def _build_cosine_mtx(cls, l_q_e, l_doc_e, emb_model):
        """
        build a q-d entity cosine similarity matrix
        :param l_q_e: query entities
        :param l_doc_e: doc entities
        :param emb_model: embedding model loaded
        :return: a matrix with cosine(q_e, doc_e)
        """
        sim_mtx = np.zeros((len(l_q_e), len(l_doc_e)))
        for i in xrange(len(l_q_e)):
            q_e = l_q_e[i]
            for j in xrange(len(l_doc_e)):
                d_e = l_doc_e[j]
                if q_e == d_e:
                    sim_mtx[i, j] = 1.0
                    continue
                if (q_e in emb_model) & (d_e in emb_model):
                    sim_mtx[i, j] = emb_model.similarity(q_e, d_e)
        return sim_mtx

    @classmethod
    def _build_l1_mtx(cls, l_q_e, l_doc_e, emb_model):
        """
        build a q-d entity l1 similarity matrix
        :param l_q_e: query entities
        :param l_doc_e: doc entities
        :param emb_model: embedding model loaded
        :return: a matrix with l1 sim(q_e, doc_e)
        """
        sim_mtx = np.zeros((len(l_q_e), len(l_doc_e)))
        for i in xrange(len(l_q_e)):
            q_e = l_q_e[i]
            for j in xrange(len(l_doc_e)):
                d_e = l_doc_e[j]
                if q_e == d_e:
                    sim_mtx[i, j] = 1.0
                    continue
                if (q_e in emb_model) & (d_e in emb_model):
                    sim_mtx[i, j] = 1.0 - np.mean(np.abs(emb_model[q_e] - emb_model[d_e]))
        return sim_mtx

    def _mean_bin(self, m_sim_mtx, l_weights=None):
        """
        return log(mean_(q_term) cosine(q_term, doc_term) bin number)
        :param m_sim_mtx: cosine similarity between q_e and d_e
        :return:
        """
        if (m_sim_mtx.shape[0] == 0) | (m_sim_mtx.shape[1] == 0):
            v_sim_vec = np.zeros(0)
        else:
            v_sim_vec = np.mean(m_sim_mtx, axis=0)
        l_bin_score = self._bin_similarity(v_sim_vec, l_weights)
        l_bin_score = [('Mean' + item[0], item[1]) for item in l_bin_score]
        return l_bin_score

    def _max_bin(self, m_sim_mtx, l_weights=None):
        """
        max bin values
        :param m_sim_mtx:
        :return:
        """
        if (m_sim_mtx.shape[0] == 0) | (m_sim_mtx.shape[1] == 0):
            v_sim_vec = np.zeros(0)
        else:
            v_sim_vec = np.max(m_sim_mtx, axis=0)
        l_bin_score = self._bin_similarity(v_sim_vec, l_weights)
        l_bin_score = [('Max' + item[0], item[1]) for item in l_bin_score]
        return l_bin_score

    def _bin_similarity(self, v_sim, l_weights=None):
        """
        :param v_sim_mtx:
        :param l_weights: the score on the corresponding entity, if empty, uniform (Frequency)
        :return: names and bin scores
        """
        l_bins = self.l_bins
        l_bin_nb = [0] * len(l_bins)
        for p in xrange(v_sim.shape[0]):
            weight = 1
            if l_weights:
                weight = l_weights[p]
            for bin_p in xrange(len(l_bins)):
                if v_sim[p] >= l_bins[bin_p]:
                    l_bin_nb[bin_p] += weight
                    break
        if self.bin_func == 'log':
            l_bin_nb = [math.log(max(score, self.log_min)) for score in l_bin_nb]
        if self.bin_func == 'norm_tf':
            z = float(sum(l_bin_nb))
            if z:
                l_bin_nb = [score / z for score in l_bin_nb]
        l_names = ['bin_%d' % i for i in xrange(len(l_bins))]
        return zip(l_names, l_bin_nb)

    # def _form_bins(self):
    #     if self.l_bins:
    #         return
    #     l_bins = [1]
    #     if self.nb_bin == 1:
    #         return l_bins
    #     bin_size = self.bin_range / (self.nb_bin - 1)
    #     for i in xrange(self.nb_bin - 1):
    #         bound = l_bins[i] - bin_size
    #         if bound == 0:
    #             bound = 0.00000001
    #         l_bins.append(bound)
    #     self.l_bins = l_bins


    # def _mean_all(self, m_sim_mtx):
    #     """
    #     total mean pool
    #     :param m_sim_mtx:
    #     :return:
    #     """
    #     score = 0
    #     l_sim = m_sim_mtx.reshape((-1,)).tolist()
    #     if l_sim:
    #         score = np.mean(m_sim_mtx)
    #     l_bin_score = [('MeanAll', score)]
    #     return l_bin_score
    #
    # def _top_k_all(self, m_sim_mtx):
    #     """
    #     total top k pool
    #     :param m_sim_mtx:
    #     :return:
    #     """
    #     l_sim = m_sim_mtx.reshape((-1,)).tolist()
    #     l_sim.sort(reverse=True)
    #     l_bin_score = []
    #     for k in xrange(self.top_k):
    #         if k < len(l_sim):
    #             l_bin_score.append(('Top_%d' % k, l_sim[k]))
    #         else:
    #             l_bin_score.append(('Top_%d' % k, 0))
    #     return l_bin_score