"""
feature extraction center

input:
    trec format candidate documents
    doc info
    q info
    feature group:
        base retrieval

output:
    svm format features

"""

import json
import logging

from traitlets import (
    Unicode,
    List,
)
from traitlets.config import Configurable

from knowledge4ir.boe_exact import (
    AnaMatch,
    CoreferenceMatch,
)
from knowledge4ir.boe_exact.salient_feature import SalientFeature
from knowledge4ir.boe_exact.nlss_feature import EGridNLSSFeature, NLSSExpansionFeature
from knowledge4ir.utils import (
    load_py_config,
    load_trec_ranking_with_score,
    load_trec_labels_dict,
    dump_svm_from_raw,
    set_basic_log,
)
from knowledge4ir.utils.resource import JointSemanticResource


class BoeLeToRFeatureExtractCenter(Configurable):
    l_feature_group = List(Unicode, default_value=['AnaExact'],
                           help="list of feature groups to use, current supporting: AnaExact,CoRef"
                           ).tag(config=True)
    h_feature_extractor = {"AnaExact": AnaMatch,
                           "CoRef": CoreferenceMatch,
                           "Salient": SalientFeature,
                           "GridNLSS": EGridNLSSFeature,
                           "NLSSExp": NLSSExpansionFeature,
                           }

    trec_rank_in = Unicode(help='trec rank candidate doc in').tag(config=True)
    q_info_in = Unicode(help='prepared query info in').tag(config=True)
    doc_info_in = Unicode(help='prepared doc info in').tag(config=True)
    qrel_in = Unicode(help='qrel').tag(config=True)
    out_name = Unicode(help='svm feature output place').tag(config=True)
    
    def __init__(self, **kwargs):
        super(BoeLeToRFeatureExtractCenter, self).__init__(**kwargs)
        self._load_data()
        self.resource = JointSemanticResource(**kwargs)
        self.l_extractor = []
        self.h_q_info = dict()
        self.h_doc_info = dict()
        self.h_qrel = dict()

        self._set_extractor(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(BoeLeToRFeatureExtractCenter, cls).class_print_help(inst)
        for name, extractor in cls.h_feature_extractor.items():
            print name
            extractor.class_print_help(inst)

    def _set_extractor(self, **kwargs):
        for name in self.l_feature_group:
            if name not in self.h_feature_extractor:
                logging.error('[%s] not in %s', name, json.dumps(self.h_feature_extractor.keys()))
            assert name in self.h_feature_extractor
            self.l_extractor.append(self.h_feature_extractor[name](**kwargs))
            logging.info('init [%s] extractor', name)
        for extractor in self.l_extractor:
            extractor.set_resource(self.resource)
        logging.info('total [%d] group features', len(self.l_feature_group))

    def _load_data(self):
        self.h_qrel = load_trec_labels_dict(self.qrel_in)
        logging.info('loaded qrel [%s]', self.qrel_in)

        logging.info('loading q info')
        l_h_data = [json.loads(line) for line in open(self.q_info_in)]
        l_qid = [h['qid'] for h in l_h_data]
        self.h_q_info = dict(zip(l_qid, l_h_data))
        logging.info('loaded [%d] q info [%s]', len(self.h_q_info), self.q_info_in)

        logging.info('loading doc info')
        # l_h_data = [json.loads(line) for line in open(self.doc_info_in)]
        # l_docno = [h['docno'] for h in l_h_data]
        self.h_doc_info = {}
        for line in open(self.doc_info_in):
            docno = json.loads(line)['docno']
            self.h_doc_info[docno] = line.strip()
        # self.h_doc_info = dict(zip(l_docno, l_h_data))
        logging.info('loaded [%d] doc info [%s]', len(self.h_doc_info), self.doc_info_in)

    def extract(self):
        l_q_rank = load_trec_ranking_with_score(self.trec_rank_in)
        l_qid = []
        l_docno = []
        l_h_feature = []
        l_label = []
        for q, ranking in l_q_rank:
            q_info = self.h_q_info[q]
            logging.info('start extracting q [%s]', q)
            for docno, base_score in ranking:
                doc_info = self.h_doc_info.get(docno, {'docno': docno})
                if type(doc_info) is str:
                    doc_info = json.loads(doc_info)
                label = self.h_qrel.get(q, {}).get(docno, 0)
                h_feature = dict()
                h_feature['base'] = base_score

                for extractor in self.l_extractor:
                    h_feature.update(extractor.extract_pair(q_info, doc_info))

                l_qid.append(q)
                l_docno.append(docno)
                l_h_feature.append(h_feature)
                l_label.append(label)
                logging.debug('[%s][%s] feature %s', q, docno, json.dumps(h_feature))

        logging.info('extraction finished, dumping...')

        h_name = dump_svm_from_raw(self.out_name, l_qid, l_docno, l_label, l_h_feature)
        logging.info('ranking features dumped to [%s]', self.out_name)
        json.dump(h_name, open(self.out_name + '_name.json', 'w'), indent=1)
        logging.info('ranking name dumped to [%s_name.json]', self.out_name)
        self._close_extractor()
        return

    def _close_extractor(self):
        for extractor in self.l_extractor:
            extractor.close_resource()


if __name__ == '__main__':
    import sys
    set_basic_log(logging.INFO)

    if 2 != len(sys.argv):
        print "1 para: config"
        BoeLeToRFeatureExtractCenter.class_print_help()
        sys.exit(-1)

    center = BoeLeToRFeatureExtractCenter(config=load_py_config(sys.argv[1]))
    center.extract()




