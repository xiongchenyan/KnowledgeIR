"""
check the bin frequency of translation matrix
input:
     translation matrix
output:
    print out the bin count

bin:
    1, (1-0.8] -> -1
"""

import json
import numpy as np
import sys

if len(sys.argv) < 2:
    print "count bin frequency"
    print "1 para: input npy translation matrix"
    sys.exit(-1)


l_bin_ed = [1.00001] + [1 - 0.2 * i for i in range(10)]
l_bin_st = [0.99999] + [ed - 0.2 for ed in l_bin_ed]
l_bin = zip(l_bin_st, l_bin_ed)
print "bins: %s" % json.dumps(l_bin)

print "loading npy"
m = np.load(sys.argv[1])
print m.shape
l_bin_cnt = []

for st, ed in l_bin:
    cnt = np.sum((m < ed) & (m >= st))
    l_bin_cnt.append(cnt)
    print "[%f, %f): %d" % (st, ed, cnt)

