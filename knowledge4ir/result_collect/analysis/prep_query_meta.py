"""
prepare query's meta data

1/3/2018
query's candidate doc's average length

input:
    trec rank candidate
    doc info

output:
    {qid: meta data:}

"""

import json
import logging
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode
)
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
)


class QueryMetaPrep(Configurable):
    doc_info = Unicode(help='doc info in').tag(config=True)
    q_info = Unicode(help='q info in').tag(config=True)
    trec_rank = Unicode(help='q trec rank candidate doc in').tag(config=True)
    out_name = Unicode(help='output location for the q stat').tag(config=True)

    def __init__(self, **kwargs):
        super(QueryMetaPrep, self).__init__(**kwargs)
        self.h_q_rank = dict()
        self.h_d_l_q = dict()    # d->[qid it appears in,]
        self.h_q_meta = dict()

        if self.trec_rank:
            self._load_candidate_doc()

    def _load_candidate_doc(self):
        l_q_rank = load_trec_ranking_with_score(self.trec_rank)
        self.h_q_rank = dict(l_q_rank)
        for q, rank in l_q_rank:
            self.h_q_meta[q] = {
                'nb_d': len(rank),
                'avg_doc_len': 0
            }
        for q, rank in l_q_rank:
            for d, score in rank:
                if d not in self.h_d_l_q:
                    self.h_d_l_q[d] = []
                self.h_d_l_q[d].append(q)
        logging.info('load candidate doc done')

    def _update_per_d(self, h_d_info):
        docno = h_d_info['docno']
        if docno not in self.h_d_l_q:
            logging.debug('doc [%s] has no q to update', docno)
            return
        l_q = self.h_d_l_q[docno]
        logging.info('update doc [%s] to q %s', docno, json.dumps(l_q))
        d_len = len(h_d_info.get('bodyText', []))
        for q in l_q:
            normalized_d_len = d_len / float(self.h_q_meta[q]['nb_d'])
            self.h_q_meta[q]['avg_doc_len'] += normalized_d_len
        return

    def process(self):

        for p, line in enumerate(open(self.doc_info)):
            if not p % 100:
                logging.info('processed [%d] doc', p)
            h_d_info = json.loads(line)
            self._update_per_d(h_d_info)

        logging.info('query meta prepared, dumping...')
        out = open(self.out_name, 'w')
        for q, h_meta in self.h_q_meta.items():
            h_meta['qid'] = q
            print >> out, json.dumps(h_meta)
        out.close()
        logging.info('results at [%s]', self.out_name)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log()

    if 2 != len(sys.argv):
        print "prep query meta, now only avg doc len"
        QueryMetaPrep.class_print_help()
        sys.exit(-1)

    preper = QueryMetaPrep(config=load_py_config(sys.argv[1]))
    preper.process()






