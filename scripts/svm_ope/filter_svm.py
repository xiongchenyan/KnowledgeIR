"""
filter svm to target q-d
"""

import json
import sys
from knowledge4ir.utils import load_svm_feature, dump_svm_feature, load_trec_ranking

if 4 != len(sys.argv):
    print "trec rank + svm in + out"
    sys.exit(-1)

l_q_rank = load_trec_ranking(sys.argv[1])
l_q_d_pair = sum(
    [zip([q] * len(rank[:100]), rank[:100]) for q, rank in l_q_rank],
    []
)
s_target = set(l_q_d_pair)
l_svm_data = load_svm_feature(sys.argv[2])

l_res = []
for svm_data in l_svm_data:
    qid = svm_data['qid']
    docno = svm_data['comment']
    if (qid, docno) in s_target:
        l_res.append(svm_data)

print "from [%d] -> [%d]" %(len(l_svm_data), len(l_res))
dump_svm_feature(l_res, sys.argv[3])


