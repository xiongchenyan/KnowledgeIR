"""
get spot's prf sentences
input:
    spotted query
    trec rank
    doc info
    qrel
output:
    pretty print:
        trec format, but score replaced with rel label
        duplicate each spot
            append the sentences that contains this spot in this doc, after #
    Json: TBD
"""


from traitlets.config import Configurable
import json
import sys
from traitlets import (
    Unicode,
    Int
)
from knowledge4ir.utils import (
    load_trec_labels_dict,
    load_py_config,
    set_basic_log,
    load_trec_ranking_with_score,
)
from nltk.tokenize import sent_tokenize
import logging
from knowledge4ir.joint import load_doc_info


class SpotSentence(Configurable):
    qrel_in = Unicode(help='qrel').tag(config=True)
    q_rank_in = Unicode().tag(config=True)
    doc_info_in = Unicode().tag(config=True)
    q_spot_in = Unicode().tag(config=True)
    out_name = Unicode().tag(config=True)

    def __init__(self, **kwargs):
        super(SpotSentence, self).__init__(**kwargs)
        self.h_qrel = {}
        self.h_q_rank = {}
        self.h_doc_info = {}
        self._load_data()

    def _load_data(self):
        logging.info('start loading data')
        self.h_qrel = load_trec_labels_dict(self.qrel_in)
        self.h_q_rank = dict(load_trec_ranking_with_score(self.q_rank_in))
        self.h_doc_info = load_doc_info(self.doc_info_in)
        logging.info('data loaded')

    def _process_one_doc(self, l_spot_name, text):
        """
        process one text
        alight texts to spots
        :param l_spot_name: target spots
        :param text: document's body text
        :return: spot -> [sentences that contain this spot]
        """

        h_spot_sent = dict([(spot, []) for spot in l_spot_name])
        text = ' '.join(text.split()).lower()
        l_sent = sent_tokenize(text)
        for sent in l_sent:
            for spot in h_spot_sent.keys():
                if (" " + spot + " ") in (" " + sent + " "):
                    h_spot_sent[spot].append(sent)

        h_cnt = dict([(item[0], len(item[1])) for item in h_spot_sent.items()])

        logging.info('doc spot sentence number: %s', json.dumps(h_cnt.items()))
        return h_spot_sent

    def process(self):
        """
        read one query info, get all spots
        get all ranked documents
        align scores
        get doc text
        call _process_one_doc
        dump
        :return:
        """
        out = open(self.out_name, 'w')
        h_q_spot_cnt = {}
        for line in open(self.q_spot_in):
            h_q = json.loads(line)
            qid = h_q['qid']
            logging.info('starting q [%s]', qid)
            l_spot = self._seg_spot(h_q)
            l_doc_score = self.h_q_rank.get(qid)
            l_doc_score = self._replace_label(qid, l_doc_score)

            for doc, score in l_doc_score:
                d_info = self.h_doc_info.get(doc, {})
                text = d_info.get('bodyText', "")
                h_spot_sent = self._process_one_doc(l_spot, text)
                l_lines = self._pretty_print(qid, h_q['query'], doc, score, h_spot_sent)
                print >> out, '\n'.join(l_lines)
                logging.info('[%s]-[%s] get [%d] spot sentences', qid, doc, len(l_lines))
                h_q_spot_cnt= self._update_spot_cnt(h_q_spot_cnt, qid, h_spot_sent)
        out.close()
        stat_out = open(self.out_name + '.stat', 'w')
        json.dump(h_q_spot_cnt, stat_out)
        stat_out.close()
        logging.info('finished')
        return

    def _update_spot_cnt(self, h_q_spot_cnt, qid, h_spot_sent):
        if qid not in h_q_spot_cnt:
            h_q_spot_cnt[qid] = dict()
        for spot, l_sent in h_spot_sent.items():
            if spot not in h_q_spot_cnt[qid]:
                h_q_spot_cnt[qid][spot] = len(l_sent)
            else:
                h_q_spot_cnt[qid][spot] += len(l_sent)
        return h_q_spot_cnt

    def _seg_spot(self, h_q):
        l_spot = [spot_info['surface'].lower() for spot_info in h_q['spot']['query']]
        return l_spot

    def _replace_label(self, qid, l_doc_score):
        l_res = []
        for doc, __ in l_doc_score:
            score = self.h_qrel.get(qid, {}).get(doc, 0)
            l_res.append((doc, score))
        l_res.sort(key=lambda item: -item[1])
        return l_res

    def _pretty_print(self, qid, query, docno, score, h_spot_sent):
        """
        form pretty print results
        :param qid:
        :param query:
        :param docno:
        :param score:
        :param h_spot_sent:
        :return:
        """
        l_res = []
        for spot, l_sent in h_spot_sent.items():
            for sent in l_sent:
                line = '\t'.join([qid, query, docno, "%d" % score, spot, sent])
                l_res.append(line)
        return l_res


if __name__ == '__main__':
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "fetch spot support sentences from PRF"
        print "1 para: config:"
        SpotSentence.class_print_help()
        sys.exit(-1)

    spot = SpotSentence(config=load_py_config(sys.argv[1]))
    spot.process()



