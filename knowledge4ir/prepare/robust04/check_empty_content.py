"""
check the number of empty content in robust04 data
input:
    raw text in column format
output:
    fraction of empty doc
"""

import sys

if 2 != len(sys.argv):
    print "check empty doc"
    print "1 para: raw doc in (docno \\t text)"
    sys.exit(-1)

cnt = 0
emp_cnt = 0
for line in open(sys.argv[1]):
    col = line.strip().split('\t')
    cnt += 1
    if (len(col) < 2):
        emp_cnt += 1


print "%d/%d" % (emp_cnt, cnt)


