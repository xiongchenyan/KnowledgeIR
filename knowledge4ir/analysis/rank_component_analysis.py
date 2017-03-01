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
from knowledge4ir.duet_feature import (
    LeToRFeatureExternalInfo,
    TermStat,

)
from knowledge4ir.utils import (
    load_query_info,
    load_trec_ranking_with_info,
    body_field,
    load_py_config,
    set_basic_log,
    text2lm,
    term2lm,
    load_trec_labels_dict,
)
import numpy as np
import json
import logging
import os


class RankComponentAna(Configurable):
    q_info_in = Unicode(help='q info in').tag(config=True)
    trec_with_info_in = Unicode(help='trec with info in').tag(config=True)
    out_dir = Unicode(help='out directory').tag(config=True)
    qrel_in = Unicode(help='qrel').tag(config=True)

    def __init__(self, **kwargs):
        super(RankComponentAna, self).__init__(**kwargs)
        self.external_info = LeToRFeatureExternalInfo(**kwargs)
        self.embedding = self.external_info.l_embedding[0]
        self.h_entity_texts = self.external_info.h_entity_texts
        self.h_field_h_df = self.external_info.h_field_h_df
        self.h_corpus_stat = self.external_info.h_corpus_stat

        self.h_q_info = load_query_info(self.q_info_in)
        self.ll_qid_ranked_doc = load_trec_ranking_with_info(self.trec_with_info_in)
        self.h_qrel = load_trec_labels_dict(self.qrel_in)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    @classmethod
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
        logging.info('start generating esr bin res')
        for qid, l_rank_doc in self.ll_qid_ranked_doc:
            h_q_info = self.h_q_info.get(qid, {})
            for doc, score, h_info in l_rank_doc:
                res = self._calc_esr_bin_per_pair(h_q_info, h_info)
                label = self.h_qrel.get(qid, {}).get(doc, 0)
                print >> out, '%s\t%s\t%d\t#\t%s' % (qid, doc, label, json.dumps(res))
            logging.info('q [%s] done', qid)
        out.close()
        logging.info('esr bin res to [%s]', out_name)
        return

    def _calc_esr_bin_per_pair(self, h_q_info, h_doc_info):
        # h_res = {}
        if 'tagme' not in h_doc_info:
            return []
        l_q_e_name = [(ana[0], ana[-1]) for ana in h_q_info['tagme']['query']]
        l_q_e_emb = []
        for e, name in l_q_e_name:
            if e not in self.embedding:
                l_q_e_emb.append(None)
                logging.warn('q entity [%s][%s] not in embedding', e, name)
            else:
                l_q_e_emb.append(self.embedding[e])

        l_d_e_name = [(ana[0], ana[-1]) for ana in h_doc_info['tagme'][body_field]]
        l_d_e_name = list(set(l_d_e_name))
        ll_e_sim = []
        for e in l_q_e_name:
            ll_e_sim.append([])
        for e, name in l_d_e_name:
            if e not in self.embedding:
                logging.warn('d e [%s][%s] not in embedding', e, name)
                continue
            d_e_emb = self.embedding[e]
            # logging.info('get doc e [%s][%s]', e, name)
            for p, q_e_emb in enumerate(l_q_e_emb):
                if q_e_emb is None:
                    continue
                l1 = 1.0 - np.mean(np.abs(q_e_emb - d_e_emb))
                # logging.info('[%s][%s] l1 distance: %f', l_q_e_name[p][1], name, l1)
                ll_e_sim[p].append((e, name, l1))
        for p in xrange(len(ll_e_sim)):
            ll_e_sim[p].sort(key = lambda item: -item[-1])

        l_res = zip(l_q_e_name, ll_e_sim)

        return l_res

    def generate_esearch_res(self):
        """
        for each q-d, calc all d's body's entity search scores as in lm_dir
        :return:
        """
        out_name = os.path.join(self.out_dir, 'esearch.json')
        out = open(out_name, 'w')
        logging.info('start generating esearch res')
        for qid, l_rank_doc in self.ll_qid_ranked_doc:
            h_q_info = self.h_q_info.get(qid, {})
            for doc, score, h_info in l_rank_doc:
                res = self._calc_esearch_per_pair(h_q_info, h_info)
                label = self.h_qrel.get(qid, {}).get(doc, 0)
                print >> out, '%s\t%s\t%d\t#\t%s' % (qid, doc, label, json.dumps(res))
            logging.info('q [%s] done', qid)
        out.close()
        logging.info('esearch res to [%s]', out_name)
        return

    def _calc_esearch_per_pair(self, h_q_info, h_doc_info):
        # h_res = {}
        if 'tagme' not in h_doc_info:
            return []
        l_e_name  = [(ana[0], ana[-1]) for ana in h_doc_info['tagme'][body_field]]
        query = h_q_info['query']
        q_lm = text2lm(query, clean=True)
        total_df, avg_len = self.h_corpus_stat[body_field]['total_df'], 100.0
        l_e_score = []
        for e, name in l_e_name:
            desp = self.h_entity_texts[e]['desp']
            e_lm = text2lm(desp, clean=True)
            term_stat = TermStat()
            term_stat.set_from_raw(q_lm, e_lm, self.h_field_h_df[body_field], total_df, avg_len)
            lm_dir = term_stat.lm_dir()
            l_e_score.append((e, name, lm_dir))

        l_e_score.sort(key=lambda item: -item[-1])
        # h_res['e_lm_dir'] = l_e_score[:10]

        return l_e_score[:10]


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


