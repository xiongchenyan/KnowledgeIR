"""
provide analysis data at query level
input:
    target method eva
    base method eva
    q info
output:
    to manual analysis data
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
)
from knowledge4ir.utils import load_gdeval_res, load_query_info
import os
import json
import logging


class FusionAnalysis(Configurable):
    target_eva = Unicode(help="target eva").tag(config=True)
    base_eva = Unicode(help='base eva').tag(config=True)
    q_info = Unicode(help='query info').tag(config=True)
    out_dir = Unicode(help='out dir').tag(config=True)
    
    def __init__(self, **kwargs):
        super(FusionAnalysis, self).__init__(**kwargs)
        self.h_q_info = load_query_info(self.q_info)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.h_q_eva = load_gdeval_res(self.target_eva)[0]
        self.h_base_q_eva = load_gdeval_res(self.base_eva)[0]

    def q_rel_ndcg_with_info(self):
        """
        dump query level ndcg and relative ndcg with q info
        :return:
        """
        out = open(self.out_dir + '/rel_ndcg', 'w')
        for qid, (ndcg, err) in self.h_q_eva.items():
            rel = ndcg - self.h_base_q_eva[qid][0]
            print >> out, qid + '\t%f\t%f\t%s' % (
                ndcg, rel, json.dumps(self.h_q_info[qid])
            )
        out.close()
        logging.info('rel ndcg dumped')

    def analysis(self):
        self.q_rel_ndcg_with_info()


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import load_py_config, set_basic_log
    set_basic_log()
    if 2 != len(sys.argv):
        print "generate analysis data"
        print "1 para: config"
        FusionAnalysis().class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])
    analyzer = FusionAnalysis(config=conf)
    analyzer.analysis()
