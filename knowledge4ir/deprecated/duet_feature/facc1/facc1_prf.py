"""
generate facc1 prf
prf score = doc score / z * tf
input:
    trec rank
    facc1 docno \t obj id
    top k (default 100)
output:
    prf rank in trec rank format
"""

from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    dump_trec_ranking_with_score,
    set_basic_log,
    term2lm,
)
import math
import sys
import logging


def load_facc1_dict(in_name):
    h_doc_ana = {}
    for line in open(in_name):
        docno, oid = line.strip().split('\t')
        if docno not in h_doc_ana:
            h_doc_ana[docno] = [oid]
        else:
            h_doc_ana[docno].append(oid)

    h_doc_olm = {}
    for docno, l_e in h_doc_ana.items():
        lm = term2lm(l_e)
        h_doc_olm[docno] = lm
    logging.info('[%d] doc facc1 dict loaded', len(h_doc_olm))
    return h_doc_olm


def prf(rank, h_doc_olm):
    rank = [(item[0], math.exp(item[1])) for item in rank]
    if not rank:
        return []
    z = sum([item[1] for item in rank])

    h_e_score = {}
    for docno, score in rank:
        if docno not in h_doc_olm:
            continue
        olm = h_doc_olm[docno]
        w = score / z
        for e, tf in olm.items():
            e_score = tf * w
            if e not in h_e_score:
                h_e_score[e] = e_score
            else:
                h_e_score[e] += e_score
    l_e_rank = h_e_score.items()
    l_e_rank.sort(key=lambda item: -item[1])
    return l_e_rank


def facc1_prf(trec_in, facc1_in, out_name):
    l_q_rank = load_trec_ranking_with_score(trec_in)
    h_doc_olm = load_facc1_dict(facc1_in)

    l_q_e_rank = []
    for q, rank in l_q_rank:
        l_e_rank = prf(rank, h_doc_olm)
        l_q_e_rank.append([q, l_e_rank])

    dump_trec_ranking_with_score(l_q_e_rank, out_name)


if __name__ == '__main__':
    set_basic_log()
    if 4 != len(sys.argv):
        print "3 para: trec rank in + facc1 prepared + prf entity out"
        sys.exit(-1)
    facc1_prf(*sys.argv[1:])

