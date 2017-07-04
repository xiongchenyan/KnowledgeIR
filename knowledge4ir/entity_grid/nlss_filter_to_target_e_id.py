"""
filer nlss to target e id
input:
    nlss, first col is key
    target id, first col is id
output:
    nlss with key in target id set
"""

import sys

if 4 != len(sys.argv):
    print "filter nlss"
    print "3 para: nlss in + target e id in + out"
    sys.exit(-1)

s_e = set([line.strip().split()[0] for line in open(sys.argv[2])])
s_find = set()
cnt = 0
out = open(sys.argv[3], 'w')
for line in open(sys.argv[1]):
    line = line.strip()
    key = line.split()[0]
    if key in s_e:
        print >> out, line
        cnt += 1
        s_find.add(key)

out.close()
print "finished [%d] nlss total, [%d/%d] found" % (cnt, len(s_find), len(s_e))

