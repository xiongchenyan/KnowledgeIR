"""
align doc info to trec
 input:
    trec rank
    doc info
output:
    add # {doc info} to each line of trec rank

"""

from knowledge4ir.utils import load_doc_info
import json
import sys


if 4 != len(sys.argv):
    print "I align doc info to trec rank"
    print "3 para: trec rank + doc info + out"
    sys.exit(-1)

h_h_doc_info = load_doc_info(sys.argv[2])

out = open(sys.argv[3], 'w')
cnt = 0
total_cnt = 0
for line in open(sys.argv[1]):
    line = line.strip()
    cols = line.split()
    docno = cols[2]
    p = int(cols[3])
    if p > 100:
        continue
    h = h_h_doc_info.get(docno, {})
    print >> out, line + '\t#\t' + json.dumps(h)
    total_cnt += 1
    if h:
        cnt += 1
out.close()
print "done, find [%d/%d]" % (cnt, total_cnt)

