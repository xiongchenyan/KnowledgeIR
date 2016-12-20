"""
embedding features for BOE
input:
    keyphrase's embedding in word2vec format
output:
    soft match features between query's BOE and doc's BOE
    E.g.
        top k cosine
        mean cosine
"""

from gensim.models import Word2Vec
import numpy as np
import json
import logging
from knowledge4ir.feature import LeToRFeatureExtractor
from traitlets import (
    Unicode,
    Int,
    List,
    Float,
)
import math


class LeToRBOEEmbFeatureExtractor(LeToRFeatureExtractor):
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)
    l_target_fields = List(Unicode,
                           default_value=[],
                           help='doc fields to use'
                           ).tag(config=True)
    embedding_in = Unicode(help='embedding data input (word2vec format) if only one',
                           ).tag(config=True)
    l_embedding_in = List(Unicode, default_value=[],
                          help="embedding data inputs, if more than one"
                          ).tag(config=True)
    l_embedding_name = List(Unicode, default_value=[],
                            help="names of corresponding embedding, if more than one"
                            ).tag(config=True)
    distance = Unicode('cos', help='distance function, cos|l1'
                       ).tag(config=True)
    # l_soft_similarities = List(Unicode, default_value=['TopK',
    #                                                    'Mean',
    #                                                    ]
    #                            ).tag(config=True)

    top_k = Int(3,
                help="number of soft match feature per paper field"
                ).tag(config=True)
    feature_name_pre = Unicode('BOEEmb')
    nb_bin = Int(5, help='number of similarity bins').tag(config=True)
    bin_func = Unicode('log',
                       help="the func to apply on bin count: log|tf|norm_tf"
                       ).tag(config=True)
    pool_func = List(Unicode, default_value=['max', 'mean', 'mean-all', 'topk'],
                     help="pooling at query entities"
                     ).tag(config=True)
    bin_range = Float(1.0, help="the bin range to keep in bin").tag(config=True)
    log_min = Float(1e-10, help='log of zero bin').tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRBOEEmbFeatureExtractor, self).__init__(**kwargs)
        if self.embedding_in:
            self.l_embedding_in.append(self.embedding_in)
            self.l_embedding_name.append('')
        if self.l_embedding_name:
            logging.info('start loading embedding %s', json.dumps(self.l_embedding_in))
            self.l_embedding = [Word2Vec.load_word2vec_format(embedding_in)
                                for embedding_in in self.l_embedding_in]
            logging.info('[%d] embedding loaded', len(self.l_embedding_in))

    def set_external_info(self, external_info):
        super(LeToRBOEEmbFeatureExtractor, self).set_external_info(external_info)
        self.l_embedding = external_info.l_embedding
        self.l_embedding_name = external_info.l_embedding_name

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = {}
        for name, embedding in zip(self.l_embedding_name, self.l_embedding):
            h_feature.update(self._extract_for_one_emb(h_q_info,
                                                       h_doc_info,
                                                       embedding,
                                                       name))
        return h_feature

    def _extract_for_one_emb(self, h_q_info, h_doc_info, emb_model, emb_name=""):
        h_feature = {}

        l_e = [ana[0] for ana in h_q_info[self.tagger]['query'] if ana[0] in emb_model]
        for field, l_ana in h_doc_info[self.tagger].items():
            if field not in self.l_target_fields:
                continue
            l_doc_e = [ana[0] for ana in l_ana if ana[0] in emb_model]
            m_sim_mtx = self._build_sim_mtx(l_e, l_doc_e, emb_model)
            # m_sim_mtx = self._build_cosine_mtx(l_e, l_doc_e, emb_model)
            # logging.debug('sim mtx: %s', np.array2string(m_sim_mtx))

            l_total_bin_score = []
            if 'mean' in self.pool_func:
                l_total_bin_score.extend(self._mean_bin(m_sim_mtx))
            if 'max' in self.pool_func:
                l_total_bin_score.extend(self._max_bin(m_sim_mtx))
            if 'mean-all' in self.pool_func:
                l_total_bin_score.extend(self._mean_all(m_sim_mtx))
            if 'topk' in self.pool_func:
                l_total_bin_score.extend(self._top_k_all(m_sim_mtx))

            for bin_name, score in l_total_bin_score:
                feature_name = '_'.join([self.feature_name_pre,
                                         emb_name,
                                         field.title(),
                                         bin_name.title()])
                h_feature[feature_name] = score

        return h_feature

    def _build_sim_mtx(self, l_q_e, l_doc_e, emb_model):
        if self.distance == 'cos':
            return self._build_cosine_mtx(l_q_e, l_doc_e, emb_model)
        if self.distance == 'l1':
            return self._build_l1_mtx(l_q_e, l_doc_e, emb_model)
        raise NotImplementedError

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
                    sim_mtx[i, j] = 1.0 - np.mean(np.abs(emb_model[q_e] - emb_model[d_e]))
        return sim_mtx

    def _soft_embedding_sim(self, m_sim_mtx):
        """
        Not in use not 11/04/2016
        soft matching weights
        exactly matched document's entities have been exclude before calculating the similarity matrix
        :param m_sim_mtx: q-e <-> doc-e similarity matrix
        :return: similarity scores as defined in self.l_soft_similarities
        """
        l_sim = []

        if 'Mean' in self.l_soft_similarities:
            l_sim.append(['Mean', np.mean(m_sim_mtx)])
        if 'TopK' in self.l_soft_similarities:
            mid = m_sim_mtx.reshape((m_sim_mtx.shape[0] * m_sim_mtx.shape[1],))
            l_top_k = mid[mid.argsort()[-self.top_k_soft:]].tolist()
            l_top_k.sort(reversed=True)
            for k in xrange(self.top_k_soft):
                l_sim.append(['Top%d' % (k + 1), l_top_k[k]])
        return l_sim

    def _mean_bin(self, m_sim_mtx):
        """
        return log(mean_(q_term) cosine(q_term, doc_term) bin number)
        :param m_sim_mtx: cosine similarity between q_e and d_e
        :return:
        """
        if (m_sim_mtx.shape[0] == 0) | (m_sim_mtx.shape[1] == 0):
            v_sim_vec = np.zeros(0)
        else:
            v_sim_vec = np.mean(m_sim_mtx, axis=0)
        l_bin_score = self._bin_similarity(v_sim_vec)
        l_bin_score = [('Mean' + item[0], item[1]) for item in l_bin_score]
        return l_bin_score

    def _max_bin(self, m_sim_mtx):
        """
        max bin values
        :param m_sim_mtx:
        :return:
        """
        if (m_sim_mtx.shape[0] == 0) | (m_sim_mtx.shape[1] == 0):
            v_sim_vec = np.zeros(0)
        else:
            v_sim_vec = np.max(m_sim_mtx, axis=0)
        l_bin_score = self._bin_similarity(v_sim_vec)
        l_bin_score = [('Max' + item[0], item[1]) for item in l_bin_score]
        return l_bin_score

    def _mean_all(self, m_sim_mtx):
        """
        total mean pool
        :param m_sim_mtx:
        :return:
        """
        score = 0
        l_sim = m_sim_mtx.reshape((-1,)).tolist()
        if l_sim:
            score = np.mean(m_sim_mtx)
        l_bin_score = [('MeanAll', score)]
        return l_bin_score

    def _top_k_all(self, m_sim_mtx):
        """
        total top k pool
        :param m_sim_mtx:
        :return:
        """
        l_sim = m_sim_mtx.reshape((-1,)).tolist()
        l_sim.sort(reverse=True)
        l_bin_score = []
        for k in xrange(self.top_k):
            if k < len(l_sim):
                l_bin_score.append(('Top_%d' % k, l_sim[k]))
            else:
                l_bin_score.append(('Top_%d' % k, 0))
        return l_bin_score

    def _bin_similarity(self, v_sim):
        """
        log bin
        :param v_sim_mtx:
        :return:
        """
        l_bins = self._form_bins()
        l_bin_nb = [0] * len(l_bins)
        for p in xrange(v_sim.shape[0]):
            for bin_p in xrange(len(l_bins)):
                if v_sim[p] >= l_bins[bin_p]:
                    l_bin_nb[bin_p] += 1
                    break
        # if tf, kept the same
        if self.bin_func == 'log':
            l_bin_nb = [math.log(max(score, self.log_min)) for score in l_bin_nb]
        elif self.bin_func == 'norm_tf':
            z = float(sum(l_bin_nb))
            if z:
                l_bin_nb = [score / z for score in l_bin_nb]
        l_names = ['bin_%d' % i for i in xrange(len(l_bins))]
        return zip(l_names, l_bin_nb)

    def _form_bins(self):
        l_bins = [1]
        if self.nb_bin == 1:
            return l_bins
        bin_size = self.bin_range / (self.nb_bin - 1)
        for i in xrange(self.nb_bin - 1):
            bound = l_bins[i] - bin_size
            if bound == 0:
                bound = 0.00000001
            l_bins.append(bound)
        # logging.info('using bin [%s]', json.dumps(l_bins))
        return l_bins


