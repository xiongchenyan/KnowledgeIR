"""
calculate the attention on spot sentence
input:
    output of spot_sentence.py
output:
    pretty print:
        trec ranking style with attention scors
    or can return a feature dict
"""

import sys
import json
from knowledge4ir.utils import (
    dump_trec_out_from_ranking_score,
    set_basic_log,
    load_py_config
)
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
)
import logging
from knowledge4ir.joint.resource import JointSemanticResource
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SpotSentAttention(Configurable):
    spot_sent_in = Unicode().tag(config=True)
    out_name = Unicode().tag(config=True)

    def __init__(self, **kwargs):
        super(SpotSentAttention, self).__init__(**kwargs)
        self.resource = JointSemanticResource(**kwargs)
        assert self.resource.embedding

    @classmethod
    def class_print_help(cls, inst=None):
        super(SpotSentAttention, cls).class_print_help(inst)
        JointSemanticResource.class_print_help(inst)

    def embedding_cosine(self, query, sent):
        q_emb = self._avg_emb(query.lower())
        sent_emb = self._avg_emb(sent.lower())
        return cosine_similarity(q_emb.reshape(1, -1), sent_emb.reshape(1, -1))

    def _avg_emb(self, text):
        l_t = [t for t in text.split() if t in self.resource.embedding]
        l_emb = [self.resource.embedding[t] for t in l_t]
        return np.mean(np.array(l_emb), axis=0)

    def generate_ranking(self):
        """
        generate pretty trec style ranking with a feature value
        :return:
        """

        l_qid, l_sentno, l_score = [], [], []

        for p, line in enumerate(open(self.spot_sent_in)):
            if not p % 100:
                logging.info('processing [%d] sentence', p)
            cols = line.strip().split('\t')
            if len(cols) != 7:
                logging.warn('[%s] cols # [%d]', line.strip(), len(cols))
                continue
            qid, query, _, _, _, sentno, sent = cols
            score = self.embedding_cosine(query, sent)
            l_qid.append(qid)
            l_sentno.append(sentno)
            l_score.append(score)

        dump_trec_out_from_ranking_score(l_qid, l_sentno, l_score, self.out_name, 'emb_cos')
        logging.info('finished')
        return


if __name__ == '__main__':
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "rank sent via cosine embedding"
        print "1 para: config"
        SpotSentAttention.class_print_help()
        sys.exit(-1)

    atter = SpotSentAttention(config=load_py_config(sys.argv[1]))
    atter.generate_ranking()


