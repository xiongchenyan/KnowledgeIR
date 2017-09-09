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


from traitlets import (
    Int, List, Unicode, Bool
)
from traitlets.config import Configurable

from knowledge4ir.duet_feature.matching.BoeEmb import LeToRBoeEmbFeatureExtractor
from knowledge4ir.duet_feature.matching.ESR import ESRFeatureExtractor
from knowledge4ir.duet_feature.matching.page_rank import PageRankFeatureExtractor
from knowledge4ir.duet_feature.matching.ir_fusion import (
    LeToRIRFusionFeatureExtractor,
)
from knowledge4ir.duet_feature.matching.les import LeToRLesFeatureExtractor
from knowledge4ir.duet_feature.matching.q_de_text import LeToRQDocETextFeatureExtractor
from knowledge4ir.duet_feature.matching.word2vec_histogram import LeToRWord2vecHistFeatureExtractor
from knowledge4ir.utils import load_json_info
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_trec_labels_dict,
    load_py_config,
    dump_svm_from_raw,
    reduce_data_to_qid,
    add_empty_zero_to_features
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
                            BoeEmb, Word2VecHist, Les, QDocEText'
                           ).tag(config=True)
    out_name = Unicode(help='feature out file name').tag(config=True)
    normalize = Bool(False, help='normalize or not (per q level normalize)').tag(config=True)
    include_base_retrieval = Bool(True, help="whether include base retrieval score as feature"
                                  ).tag(config=True)
    ext_base_rank = Unicode(help="external base rank if needed").tag(config=True)

    h_feature_extractor_map = {
        "IRFusion": LeToRIRFusionFeatureExtractor,
        "BoeEmb": LeToRBoeEmbFeatureExtractor,
        "Word2vecHist": LeToRWord2vecHistFeatureExtractor,
        "Les": LeToRLesFeatureExtractor,
        "QDocEText": LeToRQDocETextFeatureExtractor,
        "ESR": ESRFeatureExtractor,
        "PageRank": PageRankFeatureExtractor,
    }

    def __init__(self, **kwargs):
        super(LeToRFeatureExtractCenter, self).__init__(**kwargs)
        self._l_feature_extractor = []
        self.h_ext_base = {}
        self._h_qrel = dict()
        self._h_qid_q_info = dict()
        self._h_q_doc_score = dict()
        self._load_data()
        self._init_extractors(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(LeToRFeatureExtractCenter, cls).class_print_help(inst)
        for name, extractor in cls.h_feature_extractor_map.items():
            print "feature group: %s" % name
            extractor.class_print_help(inst)

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
        self._h_qid_q_info = load_json_info(self.q_info_in, key_field='qid')

        l_q_ranking_score = load_trec_ranking_with_score(self.q_doc_candidate_in)
        if self.ext_base_rank:
            l_q_ext_base = load_trec_ranking_with_score(self.ext_base_rank)
            for q, l_rank in l_q_ext_base:
                for doc, score in l_rank:
                    self.h_ext_base[q + '\t' + doc] = score
            logging.info('external base ranking scores loaded [%s]', self.ext_base_rank)
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
        for name in self.l_feature_group:
            if name not in self.h_feature_extractor_map:
                logging.error('extractor [%s] not recognized', name)
                raise NotImplementedError
            self._l_feature_extractor.append(self.h_feature_extractor_map[name](**kwargs))
            logging.info('add feature extractor [%s]', name)

    def pipe_extract(self, doc_info_in=None, out_name=None):
        """
        :return:
        """
        if not doc_info_in:
            doc_info_in = self.doc_info_in
        if not out_name:
            out_name = self.out_name
        h_doc_q_score = self._reverse_q_doc_dict()

        # for each doc_no, h_doc_info in doc data
        # if not in h_doc_q_score, then discard
        # get its qid, score, and then extract features
        # keep its features
        l_h_feature = []
        l_qid = []
        l_docno = []
        l_qrel = []
        cnt = 0
        for line in open(doc_info_in):
            h_doc_info = json.loads(line)
            docno = h_doc_info.get('docno', "")
            if docno not in h_doc_q_score:
                # not a candidate
                continue
            for qid in h_doc_q_score[docno].keys():
                logging.info('extracting [%s-%s]', qid, docno)
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

        logging.info('total [%d] pair extracted, dumping...', len(l_h_feature))
        l_h_feature = add_empty_zero_to_features(l_h_feature)
        l_qid, l_docno, l_h_feature = reduce_data_to_qid(l_qid, l_docno, l_h_feature)
        l_qrel = []
        for qid, docno in zip(l_qid, l_docno):
            l_qrel.append(self._h_qrel.get(qid, {}).get(docno, 0))
        h_feature_name = dump_svm_from_raw(out_name, l_qid, l_docno, l_qrel, l_h_feature)
        logging.info('feature extraction finished, results at [%s]', self.out_name)
        json.dump(h_feature_name, open(self.out_name + "_name.json", 'w'), indent=1)
        return

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
        h_feature = dict()
        if self.include_base_retrieval:
            h_feature['0_basescore'] = base_score  # add in the base retrieval model's score as base
        elif self.ext_base_rank:
            h_feature['0_extbase'] = self.h_ext_base.get(qid + '\t' + docno, -25.0)
        else:
            h_feature['0_bias'] = 1
        # score
        for extractor in self._l_feature_extractor:
            h_this_feature = extractor.extract(qid, docno, h_q_info, h_doc_info)
            h_feature.update(h_this_feature)
        return h_feature

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

    set_basic_log(logging.INFO)
    if 2 > len(sys.argv):
        print 'I extract features for target query doc pairs, with prepared data for q and doc, ' \
              'and qrels to fill in'
        print "1+ para: conf + doc info in (opt) + out (opt)"
        LeToRFeatureExtractCenter.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])

    extract_center = LeToRFeatureExtractCenter(config=conf)
    if len(sys.argv) > 2:
        extract_center.pipe_extract(*sys.argv[2:])
    else:
        extract_center.pipe_extract()
