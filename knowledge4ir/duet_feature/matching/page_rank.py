"""
extract page rank features using pre-trained embeddings
input:
    entity embedding

output:
    max, mean of query entity's PR score in the embedding
hyper-parameters:
    # restart probability: default 0.1
    field (default body, title is too short)

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
    Bool,
)
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    add_feature_prefix,
    term2lm,
)
from scipy import spatial


class PageRankFeatureExtractor(LeToRFeatureExtractor):
    # random_start_p = Float(0.1, help='random start prob').tag(config=True)
    l_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_steps = List(Int, default_value=[1, 5, 20]).tag(config=True)
    embedding_in = Unicode(help='word2vec formatted embedding in').tag(config=True)
    tagger = Unicode('spot')
    max_e_per_d = Int(1000, help='maximum e allowed per d').tag(config=True)
    restart = Bool(False, help='whether to restart randomly').tag(config=True)
    init_with_freq = Bool(True, help='whether initial pr score is frequency').tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRFeatureExtractor, self).__init__(**kwargs)
        if self.embedding_in:
            logging.info('start loading embedding %s', json.dumps(self.embedding_in))
            self.embedding = Word2Vec.load_word2vec_format(self.embedding_in)
            logging.info('embedding loaded')

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = dict()
        for field in self.l_fields:
            h_this_feature = self.extract_per_field(h_q_info, h_doc_info, field)
            h_feature.update(add_feature_prefix(h_this_feature, self.feature_name_pre + '_' + field))
        return h_feature

    def extract_per_field(self, h_q_info, h_doc_info, field):
        h_feature = dict()

        l_q_e = [ana['entities'][0]['id'] for ana in h_q_info[self.tagger]['query']
                 if ana['entities'][0]['id'] in self.embedding]

        l_doc_e = [ana['entities'][0]['id'] for ana in h_doc_info[self.tagger][field]
                   if ana['entities'][0]['id'] in self.embedding]

        l_doc_e, v_doc_e_w = self._filter_doc_e(l_doc_e)
        sim_mtx = self._build_translation_mtx(l_doc_e, v_doc_e_w, self.embedding)
        logging.info('random walk matrix with size [%d]', len(l_doc_e))
        if self.init_with_freq:
            v_init = v_doc_e_w
        else:
            v_init = np.ones(v_doc_e_w.shape)
        for step in self.l_steps:
            # can be optimized.. but let use this for now
            q_mean, q_max = 0, 0
            logging.info('start pr with step [%d]', step)
            if len(l_doc_e):
                v_pr = self._random_walk(sim_mtx, v_init, step)
                q_mean, q_max = self._pr_score_to_feature(l_q_e, l_doc_e, v_pr)
                logging.info('step [%d], mean max q entity pr score: [%f][%f]',
                             step, q_mean, q_max)
            h_feature["S%d_mean" % step] = q_mean
            h_feature['S%d_max' % step] = q_max
        return h_feature

    def _filter_doc_e(self, l_doc_e):
        h_doc_e_tf = term2lm(l_doc_e)
        l_doc_e_tf = sorted(h_doc_e_tf.items(), key=lambda item: -item[1])[:self.max_e_per_d]
        l_doc_e = [item[0] for item in l_doc_e_tf]
        z = float(sum([item[1] for item in l_doc_e_tf]))
        v_doc_e_w = np.array([item[1] / z for item in l_doc_e_tf])
        return l_doc_e, v_doc_e_w

    def _pr_score_to_feature(self, l_q_e, l_doc_e, v_pr_score):
        """
        pool pr score to feature
        :param l_q_e:
        :param l_doc_e:
        :param v_pr_score:
        :return:
        """
        l_q_pr = []
        l_q_p = [-1] * len(l_q_e)
        for p, q_e in enumerate(l_q_e):
            v_idx = np.zeros(v_pr_score.shape)
            for i in xrange(len(l_doc_e)):
                if l_doc_e[i] == q_e:
                    v_idx[i] = 1
                    l_q_p[p] = i
            l_q_pr.append(v_pr_score.dot(v_idx))
        if l_q_pr:
            v_q_pr = np.array(l_q_pr)
            logging.info('q e pos: %s', json.dumps(zip(l_q_e, l_q_p)))
            logging.info('q e pr score %s', json.dumps(v_q_pr.tolist()))
            return np.mean(v_q_pr), np.max(v_q_pr)
        else:
            return 0, 0

    def _build_translation_mtx(self, l_doc_e, v_doc_e_w, emb_model):
        """
        build a q-d entity cosine similarity matrix
        :param l_q_e: query entities
        :param l_doc_e: doc entities
        :param emb_model: embedding model loaded
        :return: a matrix with cosine(q_e, doc_e)
        """
        sim_mtx = np.zeros((len(l_doc_e), len(l_doc_e)))
        if l_doc_e:
            for i in xrange(len(l_doc_e)):
                e_i = l_doc_e[i]
                for j in xrange(len(l_doc_e)):
                    e_j = l_doc_e[j]
                    if e_i == e_j:
                        sim_mtx[i, j] = 1.0
                    else:
                        sim_mtx[i, j] = max(0, self.embedding.similarity(e_i, e_j))
            sim_mtx /= np.sum(sim_mtx, axis=0)
            if self.restart:
                sim_mtx = self._add_random_start_prob(sim_mtx, v_doc_e_w)
        logging.info('part of translation mtx: %s', json.dumps(sim_mtx[:10, :10].tolist()))
        return sim_mtx

    @classmethod
    def _random_walk(cls, sim_mtx, v_prob, step=1):
        res = np.array(v_prob)  # initial node weights
        logging.info('start random walk with step [%d]', step)
        for i in xrange(step):
            res = np.sum(sim_mtx * res, axis=1)
        logging.info('random walk done, pr scores: %s', json.dumps(res.tolist()))
        return res

    @classmethod
    def _add_random_start_prob(cls, sim_mtx, v_restart_prod):
        sim_mtx *= 0.9
        v_restart_prod /= float(np.sum(v_restart_prod))
        restart_mtx = v_restart_prod.reshape(v_restart_prod.shape[0], 1).dot(
            np.ones((1, v_restart_prod.shape[0])))
        sim_mtx += 0.1 * restart_mtx
        # logging.info('first rows of sim mtx %s', json.dumps(sim_mtx[:1,:].tolist()))
        z = np.sum(sim_mtx, axis=0)
        # logging.info('norm %s', json.dumps(z.tolist()))

        return sim_mtx

