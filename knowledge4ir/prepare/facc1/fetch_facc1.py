"""
fetch facc1 ana for candidate docs in given trec ranking
input:
    facc1 file
    trec rank
    top k (default 100)
output:
    docno \t obj_id from facc1
"""

import sys
from knowledge4ir.utils import load_trec_ranking_with_score


if 4 != len(sys.argv):
    print "3 para: trec rank in + facc1 file in + out"
    sys.exit(-1)


l_q_rank = load_trec_ranking_with_score(sys.argv[1])
s_docno = set(sum([[docno for docno, __ in rank] for _, rank in l_q_rank], []))

out = open(sys.argv[3], 'w')
for line in open(sys.argv[2]):
    cols = line.strip().split('\t')
    docno, obj_id = cols[0], cols[-1]
    if docno in s_docno:
        print >> out, docno + '\t' + obj_id

out.close()
print "finished"

