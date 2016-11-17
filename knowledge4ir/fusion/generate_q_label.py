"""
generate query level label based on whether a method performs better than b
input:
    eva of a
    eva of b
output:
    q \t +1/-1
"""

from knowledge4ir.utils import load_gdeval_res
import sys

if 4 != len(sys.argv):
    print "3 para: eva 1 + eva 2 + q level label (1>2 or not)"
    sys.exit(-1)

l_q_eva_a = load_gdeval_res(sys.argv[1])[0]
l_q_eva_b = load_gdeval_res(sys.argv[2])[0]
h_q_eva_b = dict(l_q_eva_b)
out = open(sys.argv[3], 'w')

pos = 0
neg = 0
for q, (ndcg, err) in l_q_eva_a:
    y = 1
    if q in h_q_eva_b:
        if ndcg < h_q_eva_b[q][0]:
            y = -1
    if y > 0:
        pos += 1
    else:
        neg += 1
    print >> q + '\t%d' % y
out.close()
print "finished with [%d] + [%d] -" % (pos, neg)
