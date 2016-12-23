"""
I am the basic functions used for LeToR feature extraction
6/14/2016 unit tested
    the minimum func of lm not yet. but will be ok
"""
import json
import logging
import math
import numpy as np
from scipy import spatial
from traitlets import (
    Int, Float, List, Unicode
)
import sys
from traitlets.config import Configurable
from knowledge4ir.utils import load_corpus_stat
from knowledge4ir.utils import TARGET_TEXT_FIELDS
from gensim.models import Word2Vec
from knowledge4ir.utils import load_trec_ranking_with_info
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


class TermStat(Configurable):
    """
    the term level statics of a query-doc field pair
    """

    def __init__(self, **kwargs):
        super(TermStat, self).__init__(**kwargs)
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

    def set_from_raw(self, h_q_terms, h_doc_terms, h_df, total_df=None, avg_doc_len=None):
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

        # logging.debug('q: %s, doc %s', json.dumps(h_q_terms),
        #               json.dumps(h_doc_terms))
        # logging.debug('stat res: %s', self.pretty_print())

        return

    def pretty_print(self):
        resp = "q_tf: %s\n" % np.array2string(self.v_q_tf, precision=4)
        resp += 'q_df: %s\n' % np.array2string(self.v_q_df, precision=4)
        resp += 'tf: %s\n' % np.array2string(self.v_tf, precision=4)
        resp += 'doc_len: %d\n' % self.doc_len
        resp += 'total_df: %d\n' % self.total_df
        resp += 'avg_doc_len: %.2f' % self.avg_doc_len
        return resp

    def mul_scores(self, lm_para=LmPara(), bm25_para=BM25Para()):
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
        l_name_score.append(['cosine', self.cosine()])
        l_name_score.append(['tf_idf', self.tf_idf()])
        l_name_score.append(['bool_and', self.bool_and()])
        l_name_score.append(['bool_or', self.bool_or()])

        return l_name_score

    def simple_scores(self):
        l_name_score = list()
        l_name_score.append(['lm', self.lm()])
        l_name_score.append(['coordinate', self.coordinate()])
        l_name_score.append(['cosine', self.cosine()])
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
        return score

    def lm_jm(self, lm_para=LmPara()):
        if self.doc_len == 0:
            return np.log(MIN_LM_SCORE)

        v_mid = self.v_tf / self.doc_len * (1 - lm_para.jm_lambda) \
                + lm_para.jm_lambda * self.v_q_df / self.total_df
        v_mid = np.maximum(v_mid, MIN_LM_SCORE)
        score = np.log(v_mid).dot(self.v_q_tf)
        return score

    def lm_twoway(self, lm_para=LmPara()):
        if self.doc_len == 0:
            return np.log(MIN_LM_SCORE)

        v_mid = (self.v_tf + lm_para.dir_mu * (self.v_q_df / self.total_df)) \
                / (self.doc_len + lm_para.dir_mu)
        v_mid = v_mid * (1 - lm_para.jm_lambda) \
                + lm_para.jm_lambda * self.v_q_df / self.total_df
        v_mid = np.maximum(v_mid, MIN_LM_SCORE)
        score = np.log(v_mid).dot(self.v_q_tf)
        return score

    def bm25(self, bm25_para=BM25Para()):
        if self.doc_len == 0:
            return 0
        v_q = self.v_q_tf / float(np.sum(self.v_q_tf))
        v_mid = self.v_tf * (bm25_para.k1 + 1) \
                / (self.v_tf + bm25_para.k1 * (1 - bm25_para.b + bm25_para.b * self.doc_len /
                                               self.avg_doc_len))

        v_mid = (self.total_df - self.v_q_df + 0.5) / (self.v_q_df + 0.5)
        v_mid = np.maximum(v_mid, 1.0)
        v_idf_q = np.log(v_mid)
        v_idf_q = np.maximum(v_idf_q, 0)
        score = v_mid.dot(v_q * v_idf_q)
        return score

    def cosine(self):
        if self.doc_len == 0:
            return 0
        if sum(self.v_tf) == 0:
            return 0
        v_q = self.v_q_tf / float(np.sum(self.v_q_tf))
        v_d = self.v_tf / float(self.doc_len)

        score = spatial.distance.cosine(v_q, v_d)
        if math.isnan(score):
            return 0
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


def fetch_doc_lm(h_doc_data, target_field):
    """
    fetch the term vectors from h_doc_data, for target_fields
    :param h_doc_data:
    :param target_field:
    :return: h_tf, h_df of doc's terms in the target field
    """
    h_tf = {}
    h_df = {}
    if 'term_vectors' in h_doc_data:
        if target_field in h_doc_data['term_vectors']:
            h_tf = dict([item[0:2] for item in h_doc_data['term_vectors'][target_field]])
            h_df = dict([(item[0], item[2]) for item in h_doc_data['term_vectors'][target_field]])

    return h_tf, h_df


def fetch_corpus_stat(h_q_data, target_field):
    l_total_stat = h_q_data['total_df'][target_field]
    total_df = l_total_stat[0]
    avg_doc_len = float(l_total_stat[1]) / total_df
    return total_df, avg_doc_len


def calc_term_stat(h_q_data, h_doc_data, target_field):
    """
    set a term stat object using data. for target fieldfield
    :param: h_q_data: the query data prepared
    :param: h_doc_data: the doc data prepared
    :param: target_field: the target field infor to fetch
    :return: a TermStat()
    """
    if target_field not in h_q_data['total_df']:
        logging.error('field [%s] not there', target_field)
        raise ValueError

    term_stat = TermStat()
    l_term = h_q_data['df'].keys()
    h_term = dict(zip(l_term, range(len(l_term))))
    d = len(l_term)
    term_stat.set_dim(d)

    l_total_stat = h_q_data['total_df'][target_field]
    term_stat.total_df = l_total_stat[0]
    term_stat.avg_doc_len = float(l_total_stat[1]) / term_stat.total_df

    for p, term in enumerate(l_term):
        q_df = 0
        if target_field in h_q_data['df'][term]:
            q_df = h_q_data['df'][term][target_field]
        term_stat.v_q_df[p] = q_df
        term_stat.v_q_tf[p] = 1

    doc_len = 0
    if 'term_vectors' in h_doc_data:
        if target_field in h_doc_data['term_vectors']:
            doc_vec = h_doc_data['term_vectors'][target_field]
            for item in doc_vec:
                term, tf = item[:2]
                doc_len += tf
                if term in h_term:
                    p = h_term[term]
                    term_stat.v_tf[p] = tf
    term_stat.doc_len = doc_len

    return term_stat


def _unit_test_set_term_stat(q_data_in, doc_data_in):
    """
    test set term stat
    :param q_data_in: the prepared query data
    :param doc_data_in: the prepared doc data
    :return: term_stat. the term statistics from
    """

    target_field = 'bodyText'

    q_line = open(q_data_in).readline()
    doc_line = open(doc_data_in).readline()

    h_q_data = json.loads(q_line.split('#')[-1])
    h_doc_data = json.loads(doc_line.split('\t')[-1])

    print 'q:\n%s' % json.dumps(h_q_data, indent=1)
    print 'doc:\n%s' % json.dumps(h_doc_data, indent=1)

    print 'calc term stat:'
    term_stat = calc_term_stat(h_q_data, h_doc_data, target_field)
    print term_stat.pretty_print()

    return term_stat


def _unit_test_models(term_stat):
    """
    unit test of scoring models
    :param term_stat: the set term_stat
    :return: print various results
    """
    print term_stat.pretty_print()

    print 'lm: %f' % term_stat.lm()
    print 'lm_dir: %f' % term_stat.lm_dir()
    print 'lm_jm: %f' % term_stat.lm_jm()
    print 'lm_twoway: %f' % term_stat.lm_twoway()
    print 'bm25: %f' % term_stat.bm25()
    print 'coor: %f' % term_stat.coordinate()
    print 'cosine: %f' % term_stat.cosine()
    print 'bool and: %f' % term_stat.bool_and()
    print 'bool or: %f' % term_stat.bool_or()
    print 'tf idf: %f' % term_stat.tf_idf()

    return


def load_entity_texts(entity_text_in):
    h = {}
    logging.info('loading entity texts from [%s]', entity_text_in)
    for line_cnt, line in enumerate(open(entity_text_in)):
        h_e = json.loads(line)
        h[h_e['id']] = h_e
        if not line_cnt % 10000:
            logging.info('loaded [%d] entities texts', line_cnt)
    logging.info('finished loading [%d] entities texts', len(h))
    return h


class LeToRFeatureExtractor(Configurable):
    """
    I am the base class for ltr feature extraction
    I take q's infor, doc's infor, candidate q-doc pairs, and qrels
    I output q-doc features's
    """
    feature_name_pre = Unicode(help='feature name pre').tag(config=True)

    def extract(self, qid, docno, h_q_info, h_doc_info):
        """
        extract features for q-doc pair
        :param qid:
        :param docno:
        :param h_q_info:
        :param h_doc_info:
        :return: h_feature
        """
        raise NotImplementedError

    def set_external_info(self, external_info):
        logging.info('setting external info with shared storage')
        return


class LeToRFeatureExternalInfo(Configurable):
    """
    load external info, to be used by extractors
    """
    corpus_stat_pre = Unicode(help="the file pre of corpus stats").tag(config=True)
    entity_text_in = Unicode(help="entity texts in").tag(config=True)
    l_text_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_embedding_in = List(Unicode, default_value=[],
                          help="embedding data inputs, if more than one"
                          ).tag(config=True)
    l_embedding_name = List(Unicode, default_value=[],
                            help="names of corresponding embedding, if more than one"
                            ).tag(config=True)
    word2vec_in = Unicode(help='word2vec in').tag(config=True)
    joint_emb_in = Unicode(help="word-entity joint embedding in").tag(config=True)
    entity_triple_in = Unicode(help="entity triple in").tag(config=True)
    prf_trec_in = Unicode(help='prf trec in').tag(config=True)

    def __init__(self, **kwargs):
        super(LeToRFeatureExternalInfo, self).__init__(**kwargs)
        logging.info('start loading external info...')
        self.h_field_h_df = {}
        if self.corpus_stat_pre:
            l_field_h_df, self.h_corpus_stat = load_corpus_stat(
                self.corpus_stat_pre, self.l_text_fields)
            self.h_field_h_df = dict(l_field_h_df)
        self.h_entity_texts = dict()
        if self.entity_text_in:
            self.h_entity_texts = load_entity_texts(self.entity_text_in)

        logging.info('start loading embedding %s', json.dumps(self.l_embedding_in))
        self.l_embedding = [Word2Vec.load_word2vec_format(embedding_in)
                            for embedding_in in self.l_embedding_in]
        logging.info('external info loaded')
        self.word2vec = None
        if self.word2vec_in:
            logging.info('loading word2vec [%s]', self.word2vec_in)
            self.word2vec = Word2Vec.load_word2vec_format(self.word2vec_in)
        self.joint_embedding = None
        if self.joint_emb_in:
            logging.info('loading joint embedding [%s]', self.joint_emb_in)
            self.joint_embedding = Word2Vec.load_word2vec_format(self.joint_emb_in)
        self.h_e_triples = {}
        if self.entity_triple_in:
            logging.info('loading entity triples [%s]', self.entity_triple_in)
            self.h_e_triples = load_packed_triples(self.entity_triple_in)
        self.ll_q_rank_info = []
        if self.prf_trec_in:
            logging.info('loading prf trec with info')
            self.ll_q_rank_info = load_trec_ranking_with_info(self.prf_trec_in)

        logging.info('external info loaded')



#
# if __name__ == '__main__':
#     import sys
#
#     if 3 != len(sys.argv):
#         print 'I unit test term stat and models'
#         print '2 para: q prepared data in + doc prepared data in'
#         sys.exit()
#
#     print 'term stat set test:'
#     term_stat = _unit_test_set_term_stat(sys.argv[1], sys.argv[2])
#
#     print 'models test:'
#     _unit_test_models(term_stat)

def load_packed_triples(entity_triple_in):
    h_e_triples = {}
    for line in open(entity_triple_in):
        h = json.loads(line)
        e = h['id']
        l_t = h['triples']
        h_e_triples[e] = l_t
    logging.info('loaded [%d] entity\' triples', len(h_e_triples))
    return h_e_triples



def load_query_info(in_name):
    """
    read what is output in batch_get_query_info
    :param in_name:
    :return:
    """
    l_lines = open(in_name).read().splitlines()
    l_vcol = [line.split('\t') for line in l_lines]
    l_qid = [vcol[0] for vcol in l_vcol]
    l_h_q_info = [json.loads(vcol[-1]) for vcol in l_vcol]
    logging.info('loaded [%d] q info', len(l_qid))
    return dict(zip(l_qid, l_h_q_info))


def load_doc_info(in_name):
    h_doc_2_info = {}
    for line in open(in_name):
        docno, data = line.strip().split('\t')
        h_info = json.loads(data)
        h_doc_2_info[docno] = h_info
    logging.info('loaded [%d] doc info', len(h_doc_2_info))
    return h_doc_2_info


def hash_feature(h_feature, h_feature_name):
    """
    hash feature to numpy array
    :param h_feature:
    :param h_feature_name:
    :return:
    """
    if not h_feature_name:
        l_name = h_feature.keys()
        l_name.sort()
        h_feature_name = dict(zip(l_name, range(len(l_name))))

    # m = np.zeros(len(h_feature_name))
    m = [0] * len(h_feature_name)
    for name, score in h_feature.items():
        if name not in h_feature_name:
            logging.error('%s not in feature name dict', name)
            raise KeyError
        f_id = h_feature_name[name]
        m[f_id] = score
    return m, h_feature_name
