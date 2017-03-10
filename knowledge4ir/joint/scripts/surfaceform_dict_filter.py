"""
only keep those appear in the data, to speed up
input:
    spotted q or doc (or both)
    sf dict
output:
    sf dict with those only appear in the spot
"""

import json
import sys

if 4 > len(sys.argv):
    print "3+ para: sf dict + out name + all spotted files"
    sys.exit(-1)

h_target = {}
h_sf_dict = json.load(open(sys.argv[1]))

for in_name in sys.argv[3:]:
    for line in open(in_name):
        h_info = json.loads(line)
        h_field_ana = h_info.get('spot', {})
        for field, l_ana in h_field_ana.items():
            for ana in l_ana:
                sf = ana.get('surface', '')
                if sf in h_sf_dict:
                    if sf not in h_target:
                        h_target[sf] = h_sf_dict[sf]

print "total [%d] sf to keep" % (len(h_target))
json.dump(h_target, open(sys.argv[2], 'w'), indent=1)
print "dumped"

