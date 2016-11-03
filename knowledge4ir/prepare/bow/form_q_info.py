"""
form query's bow q info
input:
    qid \t query
output:
    qid \t h_q_info.dumps
"""

import json
import sys


if 3 != len(sys.argv):
    print "i pack query to raw q info"
    print "2 para: qid query in + output"
    sys.exit()

out = open(sys.argv[2], 'w')

for line in open(sys.argv[1]):
    qid, q = line.strip().split('\t')

    print >> out, qid + '\t' + json.dumps({'query': q})

out.close()
print "done"
