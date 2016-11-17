"""
extract fusion feature,
    mostly for the entity space side
input:
    q level label
    q info
output:
    svm?
"""

from knowledge4ir.utils import (
    load_query_info,
    feature_hash,
)
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List,
)
import numpy as np
import json
import logging
from knowledge4ir.fusion.feature.q_ana import QAttAnaFeatureExtractor


class QAttentionFeatureExtractCenter(Configurable):
    q_info_in = Unicode(help='q info in').tag(config=True)
    out_name = Unicode(help='out name').tag(config=True)
    label_in = Unicode(help='qid label').tag(config=True)
    feature_group = List(Unicode, default_value=['q_ana'],
                         help='')

    def __init__(self, **kwargs):
        super(QAttentionFeatureExtractCenter, self).__init__(**kwargs)
        self.l_extractor = []
        self._init_extractor(**kwargs)
        self.h_label = dict([(line.split('\t')[0], int(line.strip().split('\t')[1]))
                             for line in open(self.label_in)])

    def _init_extractor(self, **kwargs):
        s_feature_group = set(self.feature_group)
        if 'q_ana' in s_feature_group:
            self.l_extractor.append(QAttAnaFeatureExtractor(**kwargs))

    @classmethod
    def class_print_help(cls, inst=None):
        super(QAttentionFeatureExtractCenter, cls).class_print_help(inst)
        print "feature group: q_ana"
        QAttAnaFeatureExtractor.class_print_help(inst)

    def pipe_extract(self, q_info_in=None, out_name=None):
        if not q_info_in:
            q_info_in = self.q_info_in
        if not out_name:
            out_name = self.out_name
        h_q_info = load_query_info(q_info_in)
        l_h_feature = []
        l_y = []
        l_q_info = h_q_info.items()
        l_q_info.sort(key=lambda item: int(item[0]))
        l_qid = []
        for qid, h_info in l_q_info:
            h_feature = self._extract(qid, h_info)
            y = 0
            if qid in self.h_label:
                y = self.h_label[qid]
            l_h_feature.append(h_feature)
            l_y.append(y)
            l_qid.append(qid)

        self._dump_feature_svm(l_y, l_h_feature, l_qid, out_name)
        logging.info('q att feature extracted to [%s]', out_name)
        return

    def _extract(self, qid, h_info):
        h_feature = {}
        for extractor in self.l_extractor:
            h_feature.update(extractor.extract(qid, h_info))
        return h_feature

    @classmethod
    def _dump_feature_svm(cls, l_y, l_h_feature, l_qid, out_name):
        l_h_hashed_feature, h_feature_name = feature_hash(l_h_feature)
        json.dump(h_feature_name, open(out_name + '_feature_name', 'w'), indent=1)
        out = open(out_name, 'w')
        for p in xrange(len(l_y)):
            l_feature = l_h_hashed_feature[p].items()
            l_feature.sort(key=lambda item: int(item[0]))
            feature_str = ' '.join(['%d:%.6f' % (item[0], item[1]) for item in l_feature])
            print >> out, '%d' % l_y[p] + " " + feature_str + " # " + l_qid[p]
        out.close()
        logging.info('[%d] data point feature dumped to [%s]', len(l_y), out_name)
        return


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log, load_py_config

    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print 'I extract attention features for query'
        QAttentionFeatureExtractCenter.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])

    extract_center = QAttentionFeatureExtractCenter(config=conf)
    extract_center.pipe_extract()








