"""
add place holder for those don't have parsed results
as empty docs
input:
    doc infos
    docno lists
output:
    doc infors with empty docno's
"""

import json
import sys
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS
)

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


if len(sys.argv) != 4:
    print "I add place holder for missing docnos"
    print "3 para: doc in + total doc no + output"
    sys.exit(-1)

l_total = open(sys.argv[2]).read().splitlines()

s_exist = set([line.split()[0] for line in open(sys.argv[1]).read().splitlines()])

out = open(sys.argv[3], 'w')
for line in open(sys.argv[1]):
    print >> out, line.strip()

h = dict()
h['tagme'] = dict()
for field in TARGET_TEXT_FIELDS:
    h[field] = ""

    h['tagme'][field] = []

s_h = json.dumps(h)
for docno in l_total:
    if docno not in s_exist:
        print >> out, docno + '\t' + s_h

out.close()
print 'done'



