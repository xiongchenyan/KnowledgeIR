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
from knowledge4ir.utils.nlp import raw_clean


class SpotSentAttention(Configurable):
    spot_sent_in = Unicode().tag(config=True)
    out_name = Unicode().tag(config=True)
    out_format = Unicode('json', help="output format: json|trec").tag(config=True)

    def __init__(self, **kwargs):
        super(SpotSentAttention, self).__init__(**kwargs)
        self.resource = JointSemanticResource(**kwargs)
        assert self.resource.embedding

    @classmethod
    def class_print_help(cls, inst=None):
        super(SpotSentAttention, cls).class_print_help(inst)
        JointSemanticResource.class_print_help(inst)

    def embedding_cosine(self, query, sent):
        q_emb = self._avg_emb(query)
        sent_emb = self._avg_emb(sent)
        if (sent_emb is None) | (q_emb is None):
            return -1
        return cosine_similarity(q_emb.reshape(1, -1), sent_emb.reshape(1, -1)).reshape(-1)[0]

    def _avg_emb(self, text):
        l_t = [t for t in text.split() if t in self.resource.embedding]
        if not l_t:
            return None
        l_emb = [self.resource.embedding[t] for t in l_t]
        return np.mean(np.array(l_emb), axis=0)

    def generate_ranking(self):
        """
        generate pretty trec style ranking with a feature value
        :return:
        """

        l_qid, l_sentno, l_score = [], [], []
        l_sent = []
        s_see_sent = set()
        for p, line in enumerate(open(self.spot_sent_in)):
            if not p % 100:
                logging.info('processing [%d] sentence', p)
            cols = line.strip().split('\t')
            if len(cols) != 7:
                logging.warn('[%s] cols # [%d]', line.strip(), len(cols))
                continue
            qid, query, _, _, _, sentno, sent = cols
            query = raw_clean(query)
            sent = raw_clean(sent)
            if len(sent.split()) > 100:
                continue
            if (len(sent.split()) - len(query.split())) < 5:
                continue
            if (qid + "\t" + sent) in s_see_sent:
                continue
            score = self.embedding_cosine(query, sent)
            l_qid.append(qid)
            l_sentno.append(sentno)
            l_score.append(score)
            l_sent.append(sent)
            s_see_sent.add(qid + '\t' + sent)  # unique
        if self.out_format == 'trec':
            dump_trec_out_from_ranking_score(l_qid, l_sentno, l_score, self.out_name, 'emb_cos', l_sent)
        if self.out_format == 'json':
            self._dump_prf_sent_json(self.out_name, l_qid, l_sentno, l_sent, l_score)
        logging.info('finished')
        return

    def _dump_prf_sent_json(self, out_name, l_qid, l_sentno, l_sent, l_score):
        # group to qid
        h_qid_sent = dict()
        for p in xrange(len(l_qid)):
            qid, sentno, sent, score =  l_qid[p], l_sentno[p], l_sent[p], l_score[p]
            if qid not in h_qid_sent:
                h_qid_sent[qid] = [(sentno, sent, score)]
            else:
                h_qid_sent[qid].append((sentno, sent, score))

        # sort each item
        # keep only top 100 to disk
        for qid in h_qid_sent.keys():
            h_qid_sent[qid].sort(key=lambda item: -item[-1])
            h_qid_sent[qid]= h_qid_sent[qid][:100]

        logging.info('qid -> prf sentences prepared')
        json.dump(h_qid_sent, open(out_name, 'w'), indent=1)
        logging.info('prf sentence json dict dumped to [%s]', out_name)


if __name__ == '__main__':
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "rank sent via cosine embedding"
        print "1 para: config"
        SpotSentAttention.class_print_help()
        sys.exit(-1)

    atter = SpotSentAttention(config=load_py_config(sys.argv[1]))
    atter.generate_ranking()


