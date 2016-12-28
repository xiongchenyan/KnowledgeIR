"""
collect esd rank baseline...
input:
    the run dir of EsdRank
out:
    EsdRank.trec
"""

import sys
import os


if 3 != len(sys.argv):
    print "2 para: EsdRank Run Dir + out name"
    sys.exit(-1)


out = open(sys.argv[2], 'w')

for qid in xrange(1, 251):
    l_docno = open(os.path.join(sys.argv[1], '%d_doc_docNo' % qid)).read().splitlines()
    l_score = open(os.path.join(sys.argv[1], '%d_prediction' % qid)).read().splitlines()
    l_score = [int(r) for r in l_score]
    l_doc_score = zip(l_docno, l_score)
    l_doc_score.sort(key=lambda item: -item[1])
    rank = 1
    for doc, score in l_doc_score:
        print >> out, '%d Q0 %s %d %d EsdRank' % (
            qid, doc, rank, score
        )
        rank += 1
out.close()
