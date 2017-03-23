"""
filter q and d info data for quick test
input:
    test trec rank
    full q info
    full d info
do:
    filter q and d infor to only those in test trec rank
"""

import json
import sys
from knowledge4ir.utils import (
    load_trec_ranking_with_score
)


if 4 != len(sys.argv):
    print "3 para: test trec in + q info or d info + out"
    sys.exit(-1)

l_q_rank = load_trec_ranking_with_score(sys.argv[1])

l_d = sum([[docno for docno, __ in rank] for __, rank in l_q_rank], [])
l_q = [q for q, __ in l_q_rank]

s_target = set(l_q + l_d)

out = open(sys.argv[3], 'w')

kept = 0
for p, line in enumerate(open(sys.argv[2])):
    if not p % 100:
        print "processed [%d] lines" % p

    h = json.loads(line)
    key = h.get('qid', '')
    if not key:
        key = h.get('docno', '')

    if key in s_target:
        print >> out, line.strip()
        kept += 1

out.close()
print "finished with [%d] kept" % (kept)





