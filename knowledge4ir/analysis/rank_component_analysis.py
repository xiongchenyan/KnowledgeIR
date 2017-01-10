"""
analysis ranking component of qw_de qe_de
input:
    q info
    trec rank with doc info
output:
    for each q-d pair:
        top k entity search score's entity (Lm dir on description)
        for each query entity, doc's entity in each transE bin
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
)
from knowledge4ir.feature import (
    LeToRFeatureExternalInfo,
    TermStat,

)
from knowledge4ir.utils import (
    load_query_info,
    load_trec_ranking_with_info,
    body_field,
    load_py_config,
    set_basic_log,
)
import numpy as np
import json
import logging
import os


class RankComponentAna(Configurable):
    q_info_in = Unicode(help='q info in').tag(config=True)
    trec_with_info_in = Unicode(help='trec with info in').tag(config=True)
    out_dir = Unicode(help='out directory').tag(config=True)

    def __init__(self, **kwargs):
        super(RankComponentAna, self).__init__(**kwargs)
        self.external_info = LeToRFeatureExternalInfo(**kwargs)
        self.embedding = self.external_info.l_embedding[0]
        self.h_entity_texts = self.external_info.h_entity_texts
        self.h_field_h_df = self.external_info.h_field_h_df
        self.h_corpus_stat = self.external_info.h_corpus_stat

        self.h_q_info = load_query_info(self.q_info_in)
        self.ll_qid_ranked_doc = load_trec_ranking_with_info(self.trec_with_info_in)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def class_print_help(cls, inst=None):
        super(RankComponentAna, cls).class_print_help(inst)
        LeToRFeatureExternalInfo.class_print_help(inst)

    def generate_esr_bin_res(self):
        """
        for each q-d, calc all its body's entity' distance with each q
        :return:

        """
        out_name = os.path.join(self.out_dir, 'esr_bin.json')
        out = open(out_name, 'w')

        for qid, l_rank_doc in self.ll_qid_ranked_doc:
            h_q_info = self.h_q_info.get(qid, {})
            for doc, score, h_info in l_rank_doc:
                h_res = self._calc_esr_bin_per_pair(h_q_info, h_info)
                print >> out, '%s\t%s\t#\t%s' % (qid, doc, json.dumps(h_res))
        out.close()
        logging.info('esr bin res to [%s]', out_name)
        return

    def _calc_esr_bin_per_pair(self, h_q_info, h_doc_info):
        h_res = {}
        return h_res

    def generate_esearch_res(self):
        """
        for each q-d, calc all d's body's entity search scores as in lm_dir
        :return:
        """
        out_name = os.path.join(self.out_dir, 'esearch.json')
        out = open(out_name, 'w')

        for qid, l_rank_doc in self.ll_qid_ranked_doc:
            h_q_info = self.h_q_info.get(qid, {})
            for doc, score, h_info in l_rank_doc:
                h_res = self._calc_esearch_per_pair(h_q_info, h_info)
                print >> out, '%s\t%s\t#\t%s' % (qid, doc, json.dumps(h_res))
        out.close()
        logging.info('esearch res to [%s]', out_name)
        return

    def _calc_esearch_per_pair(self, h_q_info, h_doc_info):
        h_res = {}
        return h_res


if __name__ == '__main__':
    import sys
    set_basic_log()

    if 2 != len(sys.argv):
        print "conf:"
        RankComponentAna.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    analyzer = RankComponentAna(config=conf)
    analyzer.generate_esr_bin_res()
    analyzer.generate_esearch_res()


