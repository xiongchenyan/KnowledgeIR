"""
extract feature for query doc pair
input:
    queries' information
    doc's information
    query-doc ranking pairs (as ranking candidates)
    qrel (to get labels)
output:
    q-doc features. in SVM format
"""

import json
import logging
import random

from traitlets import (
    Int, List, Dict, Unicode, Bool
)
from traitlets.config import Configurable

from knowledge4ir.feature.boe_embedding import LeToRBOEEmbFeatureExtractor
from knowledge4ir.feature.word2vec_histogram import LeToRWord2vecHistFeatureExtractor
from knowledge4ir.feature.les import LeToRLesFeatureExtractor
from knowledge4ir.feature.ir_fusion import (
    LeToRIRFusionFeatureExtractor,
)
from knowledge4ir.utils import load_query_info
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_trec_labels_dict,
    load_py_config,
)


class LeToRFeatureExtractCenter(Configurable):
    """
    The running pipeline class for LeToR
    """
    qrel_in = Unicode(help="q rel in").tag(config=True)
    q_info_in = Unicode(help="q information in").tag(config=True)
    doc_info_in = Unicode(help="doc information in").tag(config=True)
    q_doc_candidate_in = Unicode(help="q doc candidate in, trec format").tag(config=True)
    rank_top_k = Int(100, help="top k candidate docs to extract features").tag(config=True)
    l_feature_group = List(Unicode, default_value=['IRFusion'],
                           help='feature groups to extract: IRFusion,\
                            BoeEmb, Word2VecHist, Les'
                           ).tag(config=True)
    out_name = Unicode(help='feature out file name').tag(config=True)
    normalize = Bool(False, help='normalize or not (per q level normalize)').tag(config=True)

    _h_qrel = Dict(help='q relevance files to be loaded')
    _h_qid_q_info = Dict(help='qid to query info dict')
    _h_q_doc_score = Dict(help='query candidate documents pair -> base retrieval score')

    def __init__(self, **kwargs):
        super(LeToRFeatureExtractCenter, self).__init__(**kwargs)
        self._l_feature_extractor = []
        self._load_data()
        self._init_extractors(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(LeToRFeatureExtractCenter, cls).class_print_help(inst)
        print "Feature group: IRFusion"
        LeToRIRFusionFeatureExtractor.class_print_help(inst)
        print "Feature group: BoeEmb"
        LeToRBOEEmbFeatureExtractor.class_print_help(inst)
        print "Feature group: Word2vecHist"
        LeToRWord2vecHistFeatureExtractor.class_print_help(inst)
        print "Feature group: Les"
        LeToRLesFeatureExtractor.class_print_help(inst)
        # to add those needed the config

    def update_config(self, config):
        super(LeToRFeatureExtractCenter, self).update_config(config)
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
        if 'IRFusion' in self.l_feature_group:
            self._l_feature_extractor.append(LeToRIRFusionFeatureExtractor(**kwargs))
        if "BoeEmb" in self.l_feature_group:
            self._l_feature_extractor.append(LeToRBOEEmbFeatureExtractor(**kwargs))
        if "Word2vecHist" in self.l_feature_group:
            self._l_feature_extractor.append(LeToRWord2vecHistFeatureExtractor(**kwargs))
        if "Les" in self.l_feature_group:
            self._l_feature_extractor.append(LeToRLesFeatureExtractor(**kwargs))
        # if 'BoeLes' in self.l_feature_group:
        #     self._l_feature_extractor.append(LeToREIRFeatureExtractor(**kwargs))

    def pipe_extract(self):
        """
        :return:
        """
        h_doc_q_score = self._reverse_q_doc_dict()

        # for each doc_no, h_doc_info in doc data
        # if not in h_doc_q_score, then discard
        # get its qid, score, and then extract features
        # keep its features
        l_h_feature = []
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
                h_feature = self._extract(qid, docno, h_doc_info)
                l_qid.append(qid)
                l_docno.append(docno)
                l_h_feature.append(h_feature)
                logging.debug('[%s-%s] feature %s', qid, docno, json.dumps(h_feature))
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
        logging.info('total [%d] pair extracted, dumping...', len(l_h_feature))
        self._dump_svm_res_lines(l_qid, l_docno, l_h_feature)
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

    def _extract(self, qid, docno, h_doc_info):
        """
        get the results for qid-docno
        :param qid:
        :param docno:
        :param h_doc_info: the pre-loaded doc information, the biggest one, so it is the main stream
        :return: h_feature
        """
        h_q_info = self._h_qid_q_info[qid]

        base_score = self._h_q_doc_score[qid][docno]
        h_feature = {'0_basescore': base_score}  # add in the base retrieval model's score as base
        # score
        for extractor in self._l_feature_extractor:
            h_this_feature = extractor.extract(qid, docno, h_q_info, h_doc_info)
            h_feature.update(h_this_feature)
        return h_feature

    def _dump_svm_res_lines(self, l_qid, l_docno, l_h_feature):
        """
        output svm format results
        :param l_qid:
        :param l_docno:
        :param l_h_feature:
        :param out_name:
        :return: each line is a SVM line
        """
        logging.info('dumping [%d] feature lines', len(l_qid))
        out = open(self.out_name, 'w')

        # sort data in order
        l_qid, l_docno, l_h_feature = self._reduce_data_to_qid(l_qid, l_docno, l_h_feature)
        l_h_feature = self._add_empty_zero(l_h_feature)
        # for each line
        # get rel score
        # hash feature
        # output svm line, and append docno as comments

        if not l_qid:
            return

        # hash feature names
        l_feature_name = l_h_feature[0].keys()
        l_feature_name.sort()
        h_feature_name = dict(zip(l_feature_name, range(1, len(l_feature_name) + 1)))

        l_h_hashed_feature = []

        for h_feature in l_h_feature:
            h_hashed_feature, h_feature_name = self._hash_features(h_feature, h_feature_name)
            l_h_hashed_feature.append(h_hashed_feature)

        out = open(self.out_name, 'w')
        for i in xrange(len(l_qid)):
            qid = l_qid[i]
            docno = l_docno[i]
            h_hashed_feature = l_h_hashed_feature[i]
            rel_score = self._get_rel(qid, docno)
            print >> out, self._form_svm_line(qid, docno, rel_score, h_hashed_feature)

        out.close()
        json.dump(h_feature_name, open(self.out_name + '_feature_name', 'w'), indent=2)

        logging.info('svm type output to [%s], feature name at [%s_feature_name]',
                     self.out_name, self.out_name)

        return

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
    def _form_svm_line(qid, docno, rel_score, h_hashed_feature):
        _l = ['%d:%f' % (item[0], item[1]) for item in h_hashed_feature.items()]
        feature_str = ' '.join(_l)

        res = '%d qid:%s %s # %s' % (
            rel_score,
            qid,
            feature_str,
            docno
        )
        return res

    @staticmethod
    def _reduce_data_to_qid(l_qid, l_docno, l_h_feature):
        l_data = zip(l_qid, zip(l_docno, l_h_feature))
        random.shuffle(l_data)
        l_data.sort(key=lambda item: int(item[0]))
        l_qid = [item[0] for item in l_data]
        l_docno = [item[1][0] for item in l_data]
        l_h_feature = [item[1][1] for item in l_data]

        return l_qid, l_docno, l_h_feature

    def _get_rel(self, qid, docno):
        if qid not in self._h_qrel:
            return 0
        if docno not in self._h_qrel[qid]:
            return 0
        return self._h_qrel[qid][docno]

    @staticmethod
    def _add_empty_zero(l_h_feature):
        s_feature_name = set()
        for h_feature in l_h_feature:
            l_names = h_feature.keys()
            s_feature_name.update(l_names)

        for i in xrange(len(l_h_feature)):
            for feature in s_feature_name:
                if feature in s_feature_name:
                    if feature not in l_h_feature[i]:
                        l_h_feature[i][feature] = 0
        return l_h_feature

    @staticmethod
    def _hash_features(h_feature, h_feature_name):
        """
        transfer f-name into id's
        :param h_feature: the feature->value dict
        :param h_feature_name: feature name -> dim dict
        :return: h_hashed_feature, h_feature_name (updated)
        """
        h_hashed_feature = {}

        for name, value in h_feature.items():
            if name in h_feature_name:
                p = h_feature_name[name]
            else:
                p = len(h_feature_name) + 1
                h_feature_name[name] = p

            h_hashed_feature[p] = value

        return h_hashed_feature, h_feature_name


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log

    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print 'I extract features for target query doc pairs, with prepared data for q and doc, ' \
              'and qrels to fill in'
        LeToRFeatureExtractCenter.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])

    extract_center = LeToRFeatureExtractCenter(config=conf)
    extract_center.pipe_extract()
