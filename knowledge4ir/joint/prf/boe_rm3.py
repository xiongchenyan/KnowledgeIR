"""
RM3 model on BOE
input:
    doc infor with annotations
    trec ranking
    top k doc (default 20)
output:
    query's rm3 entities
    doc score softmax-ed
"""

from traitlets.config import Configurable
from knowledge4ir.joint import load_doc_info_json
import logging
import json
import math
from knowledge4ir.utils import (
    load_py_config,
    load_trec_ranking_with_score,
    set_basic_log,
    rm3,
    term2lm,
    dump_trec_ranking_with_score,
)
import sys
from traitlets import (
    Int,
    Unicode,
)


class BoeRm3(Configurable):
    trec_rank_in = Unicode().tag(config=True)
    doc_info_in = Unicode().tag(config=True)
    top_k_doc = Int(20).tag(config=True)
    out_name = Unicode().tag(config=True)

    def __init__(self, **kwargs):
        super(BoeRm3, self).__init__(**kwargs)
        self.l_q_rank = load_trec_ranking_with_score(self.trec_rank_in)
        self.h_doc_info = load_doc_info_json(self.doc_info_in)

    def _rm3_per_q(self, l_doc_score):
        """
        perform rm3 on on q's ranking
        :param l_doc_score: docno, ranking score
        :return:
        """
        l_doc_score = l_doc_score[:self.top_k_doc]
        z = float(sum([math.exp(score) for _, score in l_doc_score]))
        l_doc_score = [(item[0], math.exp(item[1]) / z) for item in l_doc_score]

        l_h_doc_tf = []
        for doc, _ in l_doc_score:
            doc_info = self.h_doc_info.get(doc, {})
            if not doc_info:
                l_h_doc_tf.append({})
                continue
            l_e = [item[0] for item in doc_info['tagme']['bodyText']]
            h_e_tf = term2lm(l_e)
            l_h_doc_tf.append(h_e_tf)
        l_rm3_e = rm3(l_doc_score, l_h_doc_tf, None, None, None, False)
        return l_rm3_e

    def process(self):
        ll_qid_rm3 = []
        for qid, l_doc_score in self.l_q_rank:
            l_rm3_e = self._rm3_per_q(l_doc_score)
            ll_qid_rm3.append([qid, l_rm3_e])
            logging.info('qid [%s] processed with [%d] prf entity', qid, len(l_rm3_e))
        dump_trec_ranking_with_score(ll_qid_rm3, self.out_name)
        logging.info('finished')


if __name__ == '__main__':
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "perform RM3 on BOE"
        print "1 para: config"
        BoeRm3.class_print_help()
        sys.exit(-1)

    prf_worker = BoeRm3(config=load_py_config(sys.argv[1]))
    prf_worker.process()









