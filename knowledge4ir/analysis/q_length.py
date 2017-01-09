"""
average q len (bow and boe)
q len vs relative performance?

input:
    q info
output:
    stats
"""

from knowledge4ir.utils import (
    load_query_info,
    get_rel_ndcg,
    load_gdeval_res,
)
import json
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)

import logging


def avg_len(h_q_info):
    l_bow_len = [len(h['query'].split()) for __, h in h_q_info.items()]
    l_boe_len = [len(h['tagme']['query']) for __, h in h_q_info.items()]
    return float(sum(l_bow_len)) / len(l_bow_len), float(sum(l_boe_len)) / len(l_boe_len)


def process(q_info_in, out_name):
    h_q_info = load_query_info(q_info_in)
    bow_len, boe_len = avg_len(h_q_info)
    out = open(out_name, 'w')
    print >> out, 'bow_avg_len: %f\nboe_avg_len: %f' % (bow_len, boe_len)

    out.close()


class QLenPerformanceAna(Configurable):
    q_info_in = Unicode(help='q info').tag(config=True)
    out_pre = Unicode().tag(config=True)
    base_eva_in = Unicode(help='base line eva').tag(config=True)
    eva_in = Unicode(help='eva in').tag(config=True)

    def __init__(self, **kwargs):
        super(QLenPerformanceAna, self).__init__(**kwargs)
        self.h_q_info = load_query_info(self.q_info_in)
        self.h_rel_ndcg = get_rel_ndcg(self.eva_in, self.base_eva_in)
        self.h_base_eva = dict(load_gdeval_res(self.base_eva_in, False))
        self.h_eva = dict(load_gdeval_res(self.eva_in, False))


    def avg_len(self):
        l_bow_len = [len(h['query'].split()) for __, h in self.h_q_info.items()]
        l_boe_len = [len(h['tagme']['query']) for __, h in self.h_q_info.items()]
        bow_len = float(sum(l_bow_len)) / len(l_bow_len)
        boe_len = float(sum(l_boe_len)) / len(l_boe_len)
        out = open(self.out_pre + '.avg_len', 'w')
        print >> out, 'bow_avg_len: %f\nboe_avg_len: %f' % (bow_len, boe_len)
        out.close()
        logging.info('avg len get')

    def rel_ndcg_at_len(self):
        # h_w_len_rel_ndcg = {}
        h_w_ndcg = {}
        h_w_base_ndcg = {}
        h_e_ndcg = {}
        h_e_base_ndcg = {}
        h_w_len_cnt = {}
        h_e_len_cnt = {}
        for q, h_info in self.h_q_info.items():
            bow_len = len(h_info['query'].split())
            boe_len = len(h_info['tagme']['query'])
            if bow_len not in h_w_len_cnt:
                h_w_len_cnt[bow_len] = 1
                h_w_ndcg[bow_len] = self.h_eva.get(q, [0, 0])[0]
                h_w_base_ndcg[bow_len] = self.h_base_eva.get(q, [0,0])[0]
            else:
                h_w_len_cnt[bow_len] += 1
                h_w_ndcg[bow_len] += self.h_eva.get(q, [0, 0])[0]
                h_w_base_ndcg[bow_len] += self.h_base_eva.get(q, [0,0])[0]
            if boe_len not in h_e_len_cnt:
                h_e_len_cnt[boe_len] = 1
                h_e_ndcg[boe_len] = self.h_eva.get(q, [0, 0])[0]
                h_e_base_ndcg[boe_len] = self.h_base_eva.get(q, [0,0])[0]
            else:
                h_e_len_cnt[boe_len] += 1
                h_e_ndcg[boe_len] += self.h_eva.get(q, [0, 0])[0]
                h_e_base_ndcg[boe_len] += self.h_base_eva.get(q, [0,0])[0]

        out = open(self.out_pre + '.rel_ndcg_at_len', 'w')
        print >> out, 'bow:\nlen,cnt,baseline, this_ndcg, rel'
        l_w_len = h_w_len_cnt.items()
        l_w_len.sort(key=lambda item: item[0])
        for w_len, cnt in l_w_len:
            base_ndcg = h_w_base_ndcg[w_len]
            ndcg = h_w_ndcg[w_len]
            print >> out, '%d, %d, %.4f, %.4f, %.4f' % (
                w_len,
                cnt,
                base_ndcg,
                ndcg,
                ndcg / base_ndcg - 1
            )

        print >> out, "\n\n"
        print >> out, 'boe:\nlen,cnt,baseline, this_ndcg, rel'

        l_e_len = h_e_len_cnt.items()
        l_e_len.sort(key=lambda item: item[0])
        for e_len, cnt in l_e_len:
            base_ndcg = h_e_base_ndcg[e_len]
            ndcg = h_e_ndcg[e_len]
            print >> out, '%d, %d, %.4f, %.4f, %.4f' % (
                e_len,
                cnt,
                base_ndcg,
                ndcg,
                ndcg / base_ndcg - 1
            )

        out.close()
        print "rel ndcg get"

    def process(self):
        self.avg_len()
        self.rel_ndcg_at_len()






if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    set_basic_log()
    if 2 != len(sys.argv):
        print "analysis based on query len in bow and boe"
        print "1 para: conf:"
        QLenPerformanceAna.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    analyzer = QLenPerformanceAna(config=conf)
    analyzer.process()


