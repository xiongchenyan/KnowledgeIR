"""
filter doc info to top k candidate doc
k default 100
input:
    doc info
    trec rank
    k (default 100)
output:
    smaller doc info
"""

from knowledge4ir.utils import load_trec_ranking
import json
import sys

if 4 > len(sys.argv):
    print "3 para: doc info + trec rank + out name + top k (default 100)"
    sys.exit(-1)

k = 100
if len(sys.argv) > 4:
    k = int(sys.argv[4])

l_q_rank = load_trec_ranking(sys.argv[2])
s_docno = set(sum([q_rank[1][:k] for q_rank in l_q_rank], []))
print "total [%d] target docno" % len(s_docno)

out = open(sys.argv[3], 'w')
total = 0
kept = 0
for line in open(sys.argv[1]):
    h = json.loads(line)
    total += 1
    if h['docno'] in s_docno:
        print >> out, json.dumps(h)
        kept += 1
out.close()
print "finished [%d] -> [%d] doc" % (total, kept)
