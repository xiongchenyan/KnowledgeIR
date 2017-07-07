"""
convert text field of entity in kg in the nlss format
    they are not real nlss, but just in the nls format for comparison
"""

import json
import sys

if 3 != len(sys.argv):
    print "2 para: entity text field json in + nlss format output"
    sys.exit(-1)

out = open(sys.argv[2], 'w')
for line in open(sys.argv[1]):
    h = json.loads(line)
    eid = h['id']
    desp = h['desp']
    docno = eid + '_desp'
    l_e = []
    print >> out, eid + '\t' + docno + '\t' + json.dumps(desp) + '\t' + json.dumps(l_e)

print "finished"
out.close()

