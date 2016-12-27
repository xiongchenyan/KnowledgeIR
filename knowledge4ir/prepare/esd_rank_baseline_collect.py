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
    l_docno = open(os.path.join(sys.argv[1], '%d_docno' % qid)).read().splitlines()
    l_rank = open(os.path.join(sys.argv[1], '%d_prediction').read().splitlines())

    l_doc_rank = zip(l_docno, l_rank)
    l_doc_rank.sort(key=lambda item: item[1])
    for doc, rank in l_doc_rank:
        print >> out, '%d Q0 %s %d %d EsdRank' % (
            qid, doc, rank, -rank
        )
out.close()
