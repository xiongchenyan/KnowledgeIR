"""
combine two info with same key (!)
input:
    doc info 1 + 2
output:
    update doc 1 with doc 2
"""

import json
from knowledge4ir.utils import load_json_info
import sys


if 4 != len(sys.argv):
    print "merge two info file"
    print "3 para: json info 1 + 2 + out name"
    sys.exit()

h_info_a = load_json_info(sys.argv[1], 'docno')
print "[%d] in [%s]" % (len(h_info_a), sys.argv[1])
h_info_b = load_json_info(sys.argv[2], 'docno')
print "[%d] in [%s]" % (len(h_info_b), sys.argv[2])

for key in h_info_a.keys():
    h = h_info_b.get(key, {})
    h_info_a[key].update(h)
print "combined, dumping..."
out = open(sys.argv[3], 'w')
for key, value in h_info_a.items():
    print >> out, json.dumps(value)
out.close()
print "finished"

