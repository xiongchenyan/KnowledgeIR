"""
extract histogram features between query terms and doc entities
input:
    an additional embedding of word and entity trained together

output:
    q's term <-> doc's e's embedding cosine histogram
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
)
from knowledge4ir.utils import (
    bin_similarity,
    form_bins,
    term2lm,
)


class LeToRQDocEHistFeatureExtractor(LeToRFeatureExtractor):
    l_target_fields = List(Unicode,
                           default_value=[],
                           help='doc fields to use'
                           ).tag(config=True)
    l_embedding_in = List(Unicode, default_value=[],
                          help="embedding data inputs, if more than one"
                          ).tag(config=True)
    l_embedding_name = List(Unicode, default_value=[],
                            help="names of corresponding embedding, if more than one"
                            ).tag(config=True)
    # l_soft_similarities = List(Unicode, default_value=['TopK',
    #                                                    'Mean',
    #                                                    ]
    #                            ).tag(config=True)

    top_k = Int(3,
                help="number of soft match feature per paper field"
                ).tag(config=True)
    feature_name_pre = Unicode('QDocEEmbHist')
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)
    nb_bin = Int(5, help='number of similarity bins').tag(config=True)
    bin_func = Unicode('log',
                       help="the func to apply on bin count: log|tf|norm_tf"
                       ).tag(config=True)
    pool_func = List(Unicode, default_value=['max', 'mean', 'mean-all', 'topk'],
                     help="pooling at query entities"
                     ).tag(config=True)
    bin_range = Float(1.0, help="the bin range to keep in bin").tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRQDocEHistFeatureExtractor, self).__init__(**kwargs)
        logging.info('start loading embedding %s', json.dumps(self.l_embedding_in))
        self.l_embedding = [Word2Vec.load_word2vec_format(embedding_in)
                            for embedding_in in self.l_embedding_in]
        logging.info('[%d] embedding loaded', len(self.l_embedding_in))
        self.l_bins = form_bins(self.nb_bin, self.bin_range)
        logging.info('use bins: %s', json.dumps(self.l_bins))

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
        assert self.tagger in h_doc_info
        l_w = h_q_info['query'].split()
        ll_doc_e_ana = self._fetch_doc_e_ana(h_doc_info)
        for field, l_doc_ana in zip(self.l_target_fields, ll_doc_e_ana):

            m_sim_mtx = self._build_cosine_mtx(l_w, l_doc_ana, emb_model)
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

    def _fetch_doc_e_ana(self, h_doc_info):
        ll_doc_e_ana = []
        for field in self.l_target_fields:
            l_e = []
            if field in h_doc_info[self.tagger]:
                l_e = [ana[0] for ana in h_doc_info[self.tagger][field]]
            ll_doc_e_ana.append(l_e)
        return ll_doc_e_ana

    @classmethod
    def _build_cosine_mtx(cls, l_q_w, l_doc_w, emb_model):
        """
        build a q-d entity cosine similarity matrix
        :param l_q_w: query words
        :param l_doc_w: doc words
        :param emb_model: embedding model loaded
        :return: a matrix with cosine(q_w, doc_w)
        """
        sim_mtx = np.zeros((len(l_q_w), len(l_doc_w)))
        for i in xrange(len(l_q_w)):
            q_e = l_q_w[i]
            for j in xrange(len(l_doc_w)):
                d_e = l_doc_w[j]
                if q_e == d_e:
                    sim_mtx[i, j] = 1.0
                    continue
                if (q_e in emb_model) & (d_e in emb_model):
                    sim_mtx[i, j] = emb_model.similarity(q_e, d_e)
        return sim_mtx

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
        l_bin_score = bin_similarity(v_sim_vec.tolist(), self.l_bins, self.bin_func)
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
        l_bin_score = bin_similarity(v_sim_vec.tolist(), self.l_bins, self.bin_func)
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
