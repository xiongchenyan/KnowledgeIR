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
)
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    add_feature_prefix,
)


class PageRankFeatureExtractor(LeToRFeatureExtractor):
    # random_start_p = Float(0.1, help='random start prob').tag(config=True)
    l_field = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_steps = List(Int, default_value=[1, 5, 100]).tag(config=True)
    embedding_in = Unicode(help='word2vec formatted embedding in').tag(config=True)
    tagger = Unicode('spot')

    def __init__(self, **kwargs):
        super(LeToRFeatureExtractor, self).__init__(**kwargs)
        if self.embedding_in:
            logging.info('start loading embedding %s', json.dumps(self.embedding_in))
            self.embedding = Word2Vec.load_word2vec_format(self.embedding_in)
            logging.info('embedding loaded')

    def extract(self, qid, docno, h_q_info, h_doc_info):
        h_feature = dict()
        for field in self.l_field:
            h_this_feature = self.extract_per_field(h_q_info, h_doc_info, field)
            h_feature.update(add_feature_prefix(h_this_feature, self.feature_name_pre + '_' + field))
        return h_feature

    def extract_per_field(self, h_q_info, h_doc_info, field):
        h_feature = dict()

        l_q_e = [ana['entities'][0]['id'] for ana in h_q_info[self.tagger]['query']
                 if ana['entities'][0]['id'] in self.embedding]
        l_doc_e = [ana['entities'][0]['id'] for ana in h_doc_info[self.tagger][field]
                   if ana['entities'][0]['id'] in self.embedding]

        sim_mtx = self._build_translation_mtx(l_doc_e, self.embedding)
        logging.info('random walk matrix with size [%d]', len(l_doc_e))
        v_init = np.ones(len(l_doc_e))
        for step in self.l_steps:
            # can be optimized.. but let use this for now
            q_mean, q_max = 0, 0
            if l_doc_e:
                v_pr = self._random_walk(sim_mtx, v_init, step)
                q_mean, q_max = self._pr_score_to_feature(l_q_e, l_doc_e, v_pr)
                logging.info('step [%d], mean max q entity pr score: [%f][%f]', step, q_mean, q_max)
            h_feature["S%d_mean" % step] = q_mean
            h_feature['S%d_max' % step] = q_max
        return h_feature

    def _pr_score_to_feature(self, l_q_e, l_doc_e, v_pr_score):
        """
        pool pr score to feature
        :param l_q_e:
        :param l_doc_e:
        :param v_pr_score:
        :return:
        """
        l_q_pr = []
        for q_e in l_q_e:
            v_idx = np.zeros(v_pr_score.shape)
            for i in xrange(len(l_doc_e)):
                if l_doc_e[i] == q_e:
                    v_idx[i] = 1
            l_q_pr.append(v_pr_score.dot(v_idx))
        v_q_pr = np.array(l_q_pr)
        return np.mean(v_q_pr), np.max(v_q_pr)

    @classmethod
    def _build_translation_mtx(cls, l_doc_e, emb_model):
        """
        build a q-d entity cosine similarity matrix
        :param l_q_e: query entities
        :param l_doc_e: doc entities
        :param emb_model: embedding model loaded
        :return: a matrix with cosine(q_e, doc_e)
        """
        sim_mtx = np.zeros((len(l_doc_e), len(l_doc_e)))
        for i in xrange(len(l_doc_e)):
            e_i = l_doc_e[i]
            for j in xrange(len(l_doc_e)):
                e_j = l_doc_e[j]
                if e_i == e_j:
                    sim_mtx[i, j] = 1.0
                    continue
                if (e_i in emb_model) & (e_j in emb_model):
                    sim_mtx[i, j] = max(emb_model.similarity(e_i, e_j), 0)
        col_z = np.sum(sim_mtx, axis=1)
        sim_mtx /= col_z
        return sim_mtx

    @classmethod
    def _random_walk(self, sim_mtx, v_prob, step=1):
        res = np.ones(v_prob.shape)
        for i in xrange(step):
            res = sim_mtx * v_prob
        return res

