"""
attention letor feature extract center
input:
    queries' information
    doc's information
    query-doc ranking pairs (as ranking candidates)
    qrel (to get labels)
    q-d feature, in four group:
        qw-dw
        qe-dw
        qw-de
        qe-de
    attention feature, in two group: (TBD)
        q term
        q entity
output:
    q-d features with attention on query term and entity
        in json format
"""

import json
import logging
import random

from traitlets import (
    Int, List, Dict, Unicode, Bool
)
from traitlets.config import Configurable

from knowledge4ir.feature.boe_embedding import LeToRBOEEmbFeatureExtractor
from knowledge4ir.feature.les import LeToRLesFeatureExtractor
from knowledge4ir.feature.q_de_text import LeToRQDocETextFeatureExtractorC
from knowledge4ir.feature.ir_fusion import LeToRIRFusionFeatureExtractor
from knowledge4ir.utils import load_query_info
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_trec_labels_dict,
    load_py_config,
)
import numpy as np


class AttLeToRFeatureExtractCenter(Configurable):
    """
    The running pipeline class for LeToR
    """
    qrel_in = Unicode(help="q rel in").tag(config=True)
    q_info_in = Unicode(help="q information in").tag(config=True)
    doc_info_in = Unicode(help="doc information in").tag(config=True)
    q_doc_candidate_in = Unicode(help="q doc candidate in, trec format").tag(config=True)
    rank_top_k = Int(100, help="top k candidate docs to extract features").tag(config=True)

    l_qw_dw_feature = List(Unicode, default_value=['IRFusion'],
                           help='feature between q bow d bow: IRFUsion',
                           ).tag(config=True)
    l_qw_de_feature = List(Unicode, default_value=['QDocEText'],
                           help='feature between q bow to d boe'
                           ).tag(config=True)
    l_qe_dw_feature = List(Unicode, default_value=['Les'],
                           help='feature between q boe to d bow'
                           ).tag(config=True)
    l_qe_de_feature = List(Unicode, default_value=['BoeEmb'],
                           ).tag(config=True)

    out_name = Unicode(help='feature out file name').tag(config=True)

    # normalize = Bool(False, help='normalize or not (per q level normalize)').tag(config=True)

    def __init__(self, **kwargs):
        super(AttLeToRFeatureExtractCenter, self).__init__(**kwargs)
        self._h_qrel = dict()  # q relevance files
        self._h_qid_q_info = dict()
        self._h_q_doc_score = dict()
        self.l_qw_dw_extractor = []
        self.l_qw_de_extractor = []
        self.l_qe_dw_extractor = []
        self.l_qe_de_extractor = []

        self._load_data()
        self._init_extractors(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(AttLeToRFeatureExtractCenter, cls).class_print_help(inst)
        print "Feature group: IRFusion"
        LeToRIRFusionFeatureExtractor.class_print_help(inst)
        print "Feature group: BoeEmb"
        LeToRBOEEmbFeatureExtractor.class_print_help(inst)
        print "Feature group: Les"
        LeToRLesFeatureExtractor.class_print_help(inst)
        print "Feature group: QDocEText"
        LeToRQDocETextFeatureExtractorC.class_print_help(inst)

    def update_config(self, config):
        super(AttLeToRFeatureExtractCenter, self).update_config(config)
        self._load_data()
        self._init_extractors(config=config)

    def _load_data(self):
        """
        load data from the initialized data path
        load h_qrel, h_qid_q_info, h_q_doc_score
        :return:
        """
        self._h_qrel = load_trec_labels_dict(self.qrel_in)
        self._h_qid_q_info = load_query_info(self.q_info_in)

        l_q_ranking_score = load_trec_ranking_with_score(self.q_doc_candidate_in)

        for qid, ranking_score in l_q_ranking_score:
            self._h_q_doc_score[qid] = dict(ranking_score[:self.rank_top_k])
            logging.debug('q [%s] [%d] candidate docs', qid, len(self._h_q_doc_score[qid]))
        logging.info('feature extraction data pre loaded')
        return

    def _init_extractors(self, **kwargs):
        """
        initialize extractor based configuration
        :return:
        """
        if 'IRFusion' in self.l_qw_dw_feature:
            self.l_qw_dw_extractor.append(LeToRIRFusionFeatureExtractor(**kwargs))
            logging.info('add IRFusion features to qw-dw')
        if "BoeEmb" in self.l_qe_de_feature:
            self.l_qe_de_extractor.append(LeToRBOEEmbFeatureExtractor(**kwargs))
            logging.info('add BoeEmb features to qe-de')
        if "Les" in self.l_qe_dw_feature:
            self.l_qe_dw_extractor.append(LeToRLesFeatureExtractor(**kwargs))
            logging.info('add Les features to qe-dw')
        if "QDocEText" in self.l_qw_de_feature:
            self.l_qw_de_extractor.append(LeToRQDocETextFeatureExtractorC(**kwargs))
            logging.info('add QDocE features to qw-de')

    def pipe_extract(self):
        """
        :return:
        """
        h_doc_q_score = self._reverse_q_doc_dict()

        l_features = []
        l_qid = []
        l_docno = []
        cnt = 0
        for line in open(self.doc_info_in):
            cols = line.strip().split('\t')
            docno = cols[0]
            if docno not in h_doc_q_score:
                # not a candidate
                continue
            h_doc_info = json.loads(cols[1])
            for qid in h_doc_q_score[docno].keys():
                h_q_info = self._h_qid_q_info[qid]
                l_h_qt_feature, l_h_qe_feature, l_h_qt_att_feature, l_h_qe_att_feature = self._extract(
                    qid, docno, h_q_info, h_doc_info)
                l_qid.append(qid)
                l_docno.append(docno)
                l_features.append([
                    l_h_qt_feature, l_h_qe_feature,
                    l_h_qt_att_feature, l_h_qe_att_feature,
                ])
                cnt += 1
                if 0 == (cnt % 100):
                    logging.info('extracted [%d] pair', cnt)
            del h_doc_q_score[docno]
            if len(h_doc_q_score) == 0:
                logging.info('all candidate docs extracted')
                break

        # normalize
        # if self.normalize:
        #     l_qid, l_docno, l_h_feature = self._normalize(l_qid, l_docno, l_h_feature)
        # dump results
        logging.info('total [%d] pair extracted, dumping...', len(l_features))
        self._dump_feature(l_qid, l_docno, l_features)
        logging.info('feature extraction finished, results at [%s]', self.out_name)
        return

    # def _normalize(self, l_qid, l_docno, l_h_feature):
    #     l_svm_data = [{'qid': l_qid[i], 'feature': l_h_feature[i], 'comment': l_docno[i]}
    #                   for i in xrange(len(l_qid))]
    #     l_svm_data = per_q_normalize(l_svm_data)
    #     l_qid = [data['qid'] for data in l_svm_data]
    #     l_h_feature = [data['feature'] for data in l_svm_data]
    #     l_docno = [data['comment'] for data in l_svm_data]
    #     return l_qid, l_docno, l_h_feature

    def _extract(self, qid, docno, h_q_info, h_doc_info):
        """
        get the results for qid-docno
        :param qid:
        :param docno:
        :param h_q_info: query info, pre-loaded
        :param h_doc_info: the pre-loaded doc information, the biggest one, so it is the main stream
        :return: h_feature
        """
        base_score = self._h_q_doc_score[qid][docno]

        l_h_qt_info = self._split_q_info(h_q_info, target='bow')
        logging.info('[%d] terms', len(l_h_qt_info))
        l_h_qe_info = self._split_q_info(h_q_info, target='boe')
        logging.info('[%d] entities', len(l_h_qe_info))

        l_h_qt_feature = []
        l_h_qe_feature = []
        for h_qt_info in l_h_qt_info:
            h_feature = {'0_basescore': base_score}
            for extractor in self.l_qw_dw_extractor + self.l_qw_de_extractor:
                h_this_feature = extractor.extract(qid, docno, h_qt_info, h_doc_info)
                h_feature.update(h_this_feature)
                # logging.info('[%s] feature get [%s]', extractor.feature_name_pre,
                #              json.dumps(h_this_feature))
            l_h_qt_feature.append(h_feature)
        for h_qe_info in l_h_qe_info:
            h_feature = dict()
            for extractor in self.l_qe_de_extractor + self.l_qe_dw_extractor:
                h_this_feature = extractor.extract(qid, docno, h_qe_info, h_doc_info)
                h_feature.update(h_this_feature)
                # logging.info('[%s] feature get [%s]', extractor.feature_name_pre,
                #              json.dumps(h_this_feature))
            l_h_qe_feature.append(h_feature)

        l_h_qt_att = []
        l_h_qe_att = []
        for i in xrange(len(l_h_qt_info)):
            l_h_qt_att.append({'b': 1})
        for i in xrange(len(l_h_qe_info)):
            l_h_qe_att.append({'b': 1})
        return l_h_qt_feature, l_h_qe_feature, l_h_qt_att, l_h_qe_att

    def _split_q_info(self, h_q_info, target):
        if target == 'bow':
            l_h_qt_info = []
            for t in h_q_info['query'].split():
                h = {'query': t}
                l_h_qt_info.append(h)
            return l_h_qt_info
        if target == 'boe':
            l_h_qe_info = []
            query = h_q_info['query']
            for tagger in ['tagme', 'cmns']:
                if tagger not in h_q_info:
                    continue
                l_ana = h_q_info[tagger]['query']
                for ana in l_ana:
                    h = {'query': query}
                    h[tagger] = {'query': [ana]}
                    l_h_qe_info.append(h)
            return l_h_qe_info
        raise NotImplementedError

    def _dump_feature(self, l_qid, l_docno, l_features):
        """
        align features, hash, pad, and dump
        :param l_qid:
        :param l_docno:
        :param l_features: each element is features for a pair, with four dicts:
            l_h_qt_feature, l_h_qe_feature, l_h_qt_att, l_h_qe_att,
        :return:
        """
        logging.info('dumping [%d] feature lines', len(l_qid))
        out = open(self.out_name, 'w')
        # sort data in order
        l_qid, l_docno, l_features = self._reduce_data_to_qid(l_qid, l_docno, l_features)

        l_features, h_feature_hash, h_feature_stat = self._pad_att_and_ranking_features(l_features)

        json.dump(h_feature_hash, open(self.out_name + '_feature_name', 'w'))
        json.dump(h_feature_stat, open(self.out_name + '_feature_stat', 'w'))
        logging.info('ready to dump...')
        for i in xrange(len(l_qid)):
            qid = l_qid[i]
            docno = l_docno[i]
            l_feature_mtx = l_features[i]
            rel_score = self._get_rel(qid, docno)
            h_data = dict()
            h_data['q'] = qid
            h_data['doc'] = docno
            h_data['rel'] = rel_score
            h_data['feature'] = l_feature_mtx
            print >> out, json.dumps(h_data)

        logging.info('feature dumped')
        return

    def _pad_att_and_ranking_features(self, l_features):
        """
        hash, and pad array
        :param l_features: each element is features for a pair, with four dicts:
            l_h_qt_feature, l_h_qe_feature, l_h_qt_att, l_h_qe_att,
        :return:
        """
        logging.info('start hash and pad features')

        ll_h_qt_feature = [item[0] for item in l_features]
        ll_h_qe_feature = [item[1] for item in l_features]
        ll_h_qt_att = [item[2] for item in l_features]
        ll_h_qe_att = [item[3] for item in l_features]

        l_qt_rank_mtx, h_qt_feature_hash, l_qt_stat = self._hash_and_pad_feature_matrix(ll_h_qt_feature)
        l_qe_rank_mtx, h_qe_feature_hash, l_qe_stat = self._hash_and_pad_feature_matrix(ll_h_qe_feature)
        l_qt_att_mtx, h_qt_att_hash, l_qt_att_stat = self._hash_and_pad_feature_matrix(ll_h_qt_att)
        l_qe_att_mtx, h_qe_att_hash, l_qe_att_stat = self._hash_and_pad_feature_matrix(ll_h_qe_att)

        l_new_features = []
        for p in xrange(len(l_qt_rank_mtx)):
            l_new_features.append([l_qt_rank_mtx[p], l_qe_rank_mtx[p],
                                   l_qt_att_mtx[p], l_qe_att_mtx[p]])
        h_feature_hash = {'qt_rank': h_qt_feature_hash, 'qe_rank': h_qe_feature_hash,
                          'qt_att': h_qt_att_hash, 'qe_att': h_qe_att_hash}
        h_feature_stat = {'qt_rank': l_qt_stat, 'qe_rank': l_qe_stat,
                          'qt_att': l_qt_att_stat, 'qe_att': l_qe_att_stat}
        logging.info('padding finished')
        return l_new_features, h_feature_hash, h_feature_stat

    @classmethod
    def _hash_and_pad_feature_matrix(cls, ll_h_feature):
        max_unit_dim = max([len(l_h_feature) for l_h_feature in ll_h_feature])
        s_feature_name = set()
        for l_h_feature in ll_h_feature:
            for h_feature in l_h_feature:
                s_feature_name = s_feature_name.union(set(h_feature.keys()))
        l_name = list(s_feature_name)
        l_name.sort()
        h_feature_hash = dict(zip(l_name, range(len(l_name))))
        f_dim = len(h_feature_hash)

        l_feature_mtx = []
        for l_h_feature in ll_h_feature:
            mtx = np.zeros((max_unit_dim, f_dim))
            for i, h_feature in enumerate(l_h_feature):
                for name, score in h_feature.items():
                    j = h_feature_hash[name]
                    mtx[i, j] = score
            l_feature_mtx.append(mtx.tolist())
        return l_feature_mtx, h_feature_hash, [max_unit_dim, f_dim]

    def _reverse_q_doc_dict(self):
        h_doc_q_score = {}
        pair_cnt = 0
        for q, h_doc_score in self._h_q_doc_score.items():
            for doc, score in h_doc_score.items():
                if doc not in h_doc_q_score:
                    h_doc_q_score[doc] = {}
                h_doc_q_score[doc][q] = score
                pair_cnt += 1
        logging.info('total [%d] target pair to extract', pair_cnt)
        return h_doc_q_score

    @staticmethod
    def _reduce_data_to_qid(l_qid, l_docno, l_features):
        l_data = zip(l_qid, zip(l_docno, l_features))
        random.shuffle(l_data)
        l_data.sort(key=lambda item: int(item[0]))
        l_qid = [item[0] for item in l_data]
        l_docno = [item[1][0] for item in l_data]
        l_features = [item[1][1] for item in l_data]

        return l_qid, l_docno, l_features

    def _get_rel(self, qid, docno):
        if qid not in self._h_qrel:
            return 0
        if docno not in self._h_qrel[qid]:
            return 0
        return self._h_qrel[qid][docno]


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log

    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print 'I extract attention letor features for target query doc pairs' \
              'with prepared data for q and doc, ' \
              'and qrels to fill in'
        AttLeToRFeatureExtractCenter.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])

    extract_center = AttLeToRFeatureExtractCenter(config=conf)
    extract_center.pipe_extract()
