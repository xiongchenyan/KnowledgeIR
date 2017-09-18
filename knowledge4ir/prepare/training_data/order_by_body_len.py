"""
sort data by bodyText length,
to minimize padding
"""


import sys
import json


if 3 != len(sys.argv):
    print "sort doc by bodyText annoation's length"
    print "2 para: doc info in + out"
    sys.exit()

l_h = [json.loads(line) for line in open(sys.argv[1])]

print "sorting [%d] doc" % len(l_h)
l_h.sort(key=lambda item: len(item.get('spot', {}).get('bodyText',[])))

print "sorted, will discard empty doc"
out = open(sys.argv[2], 'w')
for h in l_h:
    if not len(h.get('spot', {}).get('bodyText',[])):
        continue
    print >> out, json.dumps(h)
out.close()
print "finished"


