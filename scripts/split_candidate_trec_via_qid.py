"""
split candidate trec via qid
input:
    trec, out pre, q number per file
output:
    outpre.xx

"""

from knowledge4ir.utils import load_trec_ranking_with_score, dump_trec_ranking_with_score
import sys
import math


if 4 != len(sys.argv):
    print "3 para: trec + out pre + q per file"
    sys.exit(-1)

ll_qid_rank = load_trec_ranking_with_score(sys.argv[1])
q_per_file = int(sys.argv[3])

total_cnt = math.ceil(float(len(ll_qid_rank)) / q_per_file)
out_pre = sys.argv[2]

l_name = ['%d' % i for i in xrange(1, total_cnt + 1)]
max_len = len(l_name[-1])
l_name = [out_pre + '.' + '0' * (max_len - len(name)) + name for name in l_name]

st = 0
for name in l_name:
    dump_trec_ranking_with_score(ll_qid_rank[st: st + q_per_file], name)
    st += q_per_file

print "done"

