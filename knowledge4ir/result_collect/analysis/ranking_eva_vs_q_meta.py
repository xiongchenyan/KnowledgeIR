"""
per q ranking performances vs query meta

1/3/2018
    query meta: query's average doc len ('avg_doc_len')
    eva: in gdeval format
    target metric: NDCG
"""


from knowledge4ir.utils import (
    load_json_info,
    load_gdeval_res,
)
from knowledge4ir.result_collect.analysis.base import bin_score
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
)
import json
import logging


class RankEvaAtQMeta(Configurable):
    q_meta_in = Unicode(help='meta info for q').tag(config=True)
    nb_bin = Int(10).tag(config=True)
    # eva_in = Unicode(help='gdeval results').tag(config=True)

    def __init__(self, **kwargs):
        super(RankEvaAtQMeta, self).__init__(**kwargs)
        assert self.q_meta_in
        self.h_q_meta = load_json_info(self.q_meta_in, key_field='qid')

    def process(self, eva_in, out_name):
        l_q_eva = load_gdeval_res(eva_in, with_mean=False)
        l_avg_doc_len = []
        l_ndcg = []
        for q, eva in l_q_eva:
            if not q in self.h_q_meta:
                logging.warn('q [%s] has no meta data', q)
            l_ndcg.append(eva[0])
            l_avg_doc_len.append(self.h_q_meta[q]['avg_doc_len'])

        l_bin_res, l_bin_range = bin_score(l_avg_doc_len, l_ndcg, self.nb_bin)
        h_res = {
            'avg_doc_len_bin': l_bin_res,
            'avg_doc_len_bin_rage': l_bin_range,
        }
        json.dump(h_res, open(out_name, 'w'), indent=1)
        logging.info('finished, results at [%s]', out_name)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log()
    if 4 != len(sys.argv):
        print "3 para: config + eva in + out"
        RankEvaAtQMeta.class_print_help()
        sys.exit(-1)

    aligner = RankEvaAtQMeta(config=load_py_config(sys.argv[1]))
    aligner.process(*sys.argv[2:])












