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

import numpy as np
from traitlets import (
    Int, List, Unicode
)
from traitlets.config import Configurable

from knowledge4ir.duet_feature import LeToRFeatureExternalInfo
from knowledge4ir.duet_feature.attention.e_ambiguity import EntityAmbiguityAttentionFeature
from knowledge4ir.duet_feature.attention.e_embedding import EntityEmbeddingAttentionFeature
from knowledge4ir.duet_feature.attention.e_memory import EntityMemoryAttentionFeature
from knowledge4ir.duet_feature.matching.ESR import ESRFeatureExtractor
from knowledge4ir.duet_feature.matching.ir_fusion import LeToRIRFusionFeatureExtractor
from knowledge4ir.duet_feature.matching.les import LeToRLesFeatureExtractor
from knowledge4ir.duet_feature.matching.pre_trained import LeToRBOEPreTrainedFeatureExtractor
from knowledge4ir.duet_feature.matching.q_de_text import LeToRQDocETextFeatureExtractor
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_trec_labels_dict,
    load_py_config,
    load_json_info,
    group_data_to_qid,
)


class AttLeToRFeatureExtractCenter(Configurable):
    """
    The running pipeline class for LeToR
    """
    h_match_feature_extractor = {
        "IRFusion": LeToRIRFusionFeatureExtractor,
        "Les": LeToRLesFeatureExtractor,
        "QDocEText": LeToRQDocETextFeatureExtractor,
        "ESR": ESRFeatureExtractor,
        "Pretrain": LeToRBOEPreTrainedFeatureExtractor,
    }
    h_att_feature_extractor = {
        "Emb": EntityEmbeddingAttentionFeature,
        "Mem": EntityMemoryAttentionFeature,
        "Ambi": EntityAmbiguityAttentionFeature,
    }

    qrel_in = Unicode(help="q rel in").tag(config=True)
    q_info_in = Unicode(help="q information in").tag(config=True)
    doc_info_in = Unicode(help="doc information in").tag(config=True)
    q_doc_candidate_in = Unicode(help="q doc candidate in, trec format").tag(config=True)
    rank_top_k = Int(100, help="top k candidate docs to extract features").tag(config=True)

    l_qe_att_feature = List(Unicode, default_value=['Emb'],
                            help='q e attention feature: Emb, Text, Static, Prf, Mem, Surface, Linker, Ambi'
                            ).tag(config=True)
    l_qw_match_feature = List(Unicode, default_value=['IRFusion', 'QDocEText'],
                              help='match features from qw: IRFusion',
                              ).tag(config=True)
    l_qe_match_feature = List(Unicode, default_value=['ESR', 'Les'],
                              help='match features from qe: ESR|BoeEmb'
                              ).tag(config=True)

    out_name = Unicode(help='feature out file name').tag(config=True)

    def __init__(self, **kwargs):
        super(AttLeToRFeatureExtractCenter, self).__init__(**kwargs)
        self._h_qrel = dict()  # q relevance files
        self._h_qid_q_info = dict()
        self._h_q_doc_score = dict()
        self.l_qe_att_extractor = []
        self.l_qw_match_extractor = []
        self.l_qe_match_extractor = []

        self._load_data()
        self.external_info = LeToRFeatureExternalInfo(**kwargs)
        self._init_extractors(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(AttLeToRFeatureExtractCenter, cls).class_print_help(inst)
        print "external info:"
        LeToRFeatureExternalInfo.class_print_help(inst)

        for key, extractor in cls.h_match_feature_extractor.items():
            print "match extractor: %s" % key
            extractor.class_print_help()
        for key, extractor in cls.h_att_feature_extractor.items():
            print "entity attention extractor: %s" % key
            extractor.class_print_help()

    def update_config(self, config):
        logging.info("update config")
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
        self._h_qid_q_info = load_json_info(self.q_info_in, 'qid')

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
        for extract_name in self.l_qw_match_feature:
            extractor = self.h_match_feature_extractor[extract_name](**kwargs)
            extractor.set_external_info(self.external_info)
            self.l_qw_match_extractor.append(extractor)
            logging.info('initialized [%s] extractor', extract_name)

        for extract_name in self.l_qe_match_feature:
            extractor = self.h_match_feature_extractor[extract_name](**kwargs)
            extractor.set_external_info(self.external_info)
            self.l_qe_match_extractor.append(extractor)
            logging.info('initialized [%s] extractor', extract_name)

        for extract_name in self.l_qe_att_feature:
            extractor = self.h_att_feature_extractor[extract_name](**kwargs)
            extractor.set_external_info(self.external_info)
            self.l_qe_att_extractor.append(extractor)
            logging.info('initialized [%s] extractor', extract_name)

    def pipe_extract(self, doc_info_in=None, out_name=None):
        """
        :return:
        """
        if not doc_info_in:
            doc_info_in = self.doc_info_in
        if not out_name:
            out_name = self.out_name
        h_doc_q_score = self._reverse_q_doc_dict()

        l_features = []
        l_qid = []
        l_docno = []
        cnt = 0
        for line in open(doc_info_in):
            h_doc_info = json.loads(line)
            docno = h_doc_info['docno']
            if docno not in h_doc_q_score:
                continue
            for qid in h_doc_q_score[docno].keys():
                h_q_info = self._h_qid_q_info[qid]
                # l_h_qt_feature, l_h_qe_feature, l_h_qt_att_feature, l_h_qe_att_feature \
                l_this_q_all_features = self._extract(qid, docno, h_q_info, h_doc_info)
                l_qid.append(qid)
                l_docno.append(docno)
                l_features.append(l_this_q_all_features)
                cnt += 1
                if 0 == (cnt % 100):
                    logging.info('extracted [%d] pair', cnt)
            del h_doc_q_score[docno]
            if len(h_doc_q_score) == 0:
                logging.info('all candidate docs extracted')
                break

        logging.info('total [%d] pair extracted, dumping...', len(l_features))
        self._dump_feature(l_qid, l_docno, l_features, out_name)
        logging.info('feature extraction finished, results at [%s]', out_name)
        return

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
        h_q_info['qid'] = qid

        l_h_qt_info, l_t = self._split_q_info(h_q_info, target='bow')
        l_h_qe_info, l_e = self._split_q_info(h_q_info, target='boe')

        l_h_qt_feature = []
        l_h_qe_feature = []
        for h_qt_info in l_h_qt_info:
            h_feature = {'0_basescore': base_score}
            for extractor in self.l_qw_match_extractor:
                h_this_feature = extractor.extract(qid, docno, h_qt_info, h_doc_info)
                h_feature.update(h_this_feature)
            l_h_qt_feature.append(h_feature)

        for h_qe_info in l_h_qe_info:
            h_feature = {'bias': 1}
            for extractor in self.l_qe_match_extractor:
                logging.debug('extracting [%s] qe match feature', extractor.feature_name_pre)
                h_this_feature = extractor.extract(qid, docno, h_qe_info, h_doc_info)
                h_feature.update(h_this_feature)
            l_h_qe_feature.append(h_feature)

        l_h_qt_att = []
        l_h_qe_att = []
        for i in xrange(len(l_h_qt_info)):
            l_h_qt_att.append({'b': 1})
        for i in xrange(len(l_h_qe_info)):
            l_h_qe_att.append({'b': 1})

        # for extractor in self.l_qt_att_extractor:
        #     l_h_feature = extractor.extract(h_q_info, l_t)
        #     for i in xrange(len(l_h_feature)):
        #         l_h_qt_att[i].update(l_h_feature[i])

        for extractor in self.l_qe_att_extractor:
            l_h_feature = extractor.extract(h_q_info, l_e)
            logging.debug('extracting [%s] qe att feature', extractor.feature_name_pre)
            for i in xrange(len(l_h_feature)):
                l_h_qe_att[i].update(l_h_feature[i])

        logging.info('[%s-%s] get [%d,%d,%d,%d] feature numbers',
                     qid, docno,
                     len(l_h_qt_feature),
                     len(l_h_qe_feature),
                     len(l_h_qt_att),
                     len(l_h_qe_att))
        return [l_h_qt_feature, l_h_qe_feature, l_h_qt_att, l_h_qe_att]

    def _split_q_info(self, h_q_info, target):
        if target == 'bow':
            l_t = []
            l_h_qt_info = []
            for t in h_q_info['query'].split():
                h = {'query': t}
                l_h_qt_info.append(h)
                l_t.append(t)
            return l_h_qt_info, l_t
        if target == 'boe':
            l_h_qe_info = []
            l_e = []
            query = h_q_info['query']
            for tagger in ['tagme', 'cmns']:
                if tagger not in h_q_info:
                    continue
                l_ana = h_q_info[tagger]['query']
                for ana in l_ana:
                    h = {'query': query}
                    h[tagger] = {'query': [ana]}
                    l_h_qe_info.append(h)
                    l_e.append(ana[0])
            return l_h_qe_info, l_e
        raise NotImplementedError

    def _dump_feature(self, l_qid, l_docno, l_features, out_name=None):
        """
        align features, hash, pad, and dump
        :param l_qid:
        :param l_docno:
        :param l_features: each element is features for a pair, with four dicts:
            l_h_qt_feature, l_h_qe_feature, l_h_qt_att, l_h_qe_att,
        :return:
        """
        if not out_name:
            out_name = self.out_name
        logging.info('dumping [%d] feature lines', len(l_qid))
        out = open(out_name, 'w')
        # sort data in order
        l_qid, l_docno, l_features = group_data_to_qid(l_qid, l_docno, l_features)

        l_features, h_feature_hash, h_feature_stat = self._pad_att_and_ranking_features(l_features)

        json.dump(h_feature_hash, open(out_name + '_feature_name', 'w'))
        json.dump(h_feature_stat, open(out_name + '_feature_stat', 'w'))
        logging.info('ready to dump...')
        for i in xrange(len(l_qid)):
            qid = l_qid[i]
            docno = l_docno[i]
            l_feature_mtx = l_features[i]
            rel_score = self._h_qrel.get(qid, {}).get(docno, 0)
            h_data = {
                'q': qid,
                'doc': docno,
                'rel': rel_score,
                'feature': l_feature_mtx
            }
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


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log

    set_basic_log(logging.DEBUG)
    if 2 > len(sys.argv):
        print 'I extract attention letor features for target query doc pairs' \
              'with prepared data for q and doc, ' \
              'and qrels to fill in'
        print "1+ para: conf + doc info in (opt) + out (opt)"
        AttLeToRFeatureExtractCenter.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])

    extract_center = AttLeToRFeatureExtractCenter(config=conf)
    if len(sys.argv) > 2:
        extract_center.pipe_extract(*sys.argv[2:])
    else:
        extract_center.pipe_extract()
