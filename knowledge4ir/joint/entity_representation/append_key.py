"""
append key
input:
    spot info
    boe info
do:
    the two inputs are 1-1 corresponding
    add docno or qid from spot info to boe info
    check if title or query field is the same
"""

import json
import sys

if 4 != len(sys.argv):
    print "3 para: spot info + boe info + output"
    print 'add key field'
    sys.exit(-1)

l_key = ['qid', 'docno']
l_field = ['query', 'title']

out = open(sys.argv[3], 'w')

print "loading [%s]" % sys.argv[1]
l_h = []
for line in open(sys.argv[1]):
    h = json.loads(line)
    h_slim = {}
    for key in l_key + l_field:
        if key in h:
            h_slim[key] = h[key]
    l_h.append(h_slim)

print "loading [%s]" % sys.argv[2]
for p, line in enumerate(open(sys.argv[2])):
    h = json.loads(line)
    for field in l_field:
        assert h.get(field, "") == l_h[p].get(field, "")
    for key in l_key:
        if key in l_h[p]:
            h[key] = l_h[p][key]
    print >> out, json.dumps(h)
out.close()
print "done"


