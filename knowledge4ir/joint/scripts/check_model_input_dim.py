"""
check model input's dimension
assert if there is any difference
"""


import numpy as np
import json
import sys


if 3 != len(sys.argv):
    print "2 para: model input + shape output"
    sys.exit(-1)


h_name_shape = dict()

for p, line in enumerate(open(sys.argv[1])):
    h = json.loads(line)
    for key, ts in h.items():
        m = np.array(ts)
        if key not in h_name_shape:
            h_name_shape[key] = m.shape
        else:
            if h_name_shape[key] != m.shape:
                print "%s shape [%s] error" % (json.dumps(h['meta']), json.dumps(m.shape))

json.dump(h_name_shape, open(sys.argv[2], 'w'), indent=1)
print "finished"

