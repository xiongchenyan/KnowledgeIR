"""
split the qid property in config
200 -> 20 * 10 file
"""

import sys
import json


if 3 != len(sys.argv):
    print "2 para: root conf + nb q per conf"
    sys.exit(-1)

l_conf = open(sys.argv[1]).read().splitlines()

q_range = None
q_st = 0
q_ed = 0
for line in l_conf:
    if line.startswith("c.CrossValidator.q_range="):
        q_range = line.replace("c.CrossValidator.q_range=", "")
        q_st, q_ed = q_range.strip('[').strip(']').split(',')
        q_st = int(q_st)
        q_ed = int(q_ed)
        break
print "root range %d, %d" % (q_st, q_ed)

nb_q = int(sys.argv[2])

st = q_st
ed = st + nb_q - 1
cnt = 0
while ed <= q_ed:

    with open(sys.argv[1] + "_%d" % cnt, 'w') as out:
        for line in l_conf:
            if line.startswith("c.CrossValidator.q_range="):
                print >> out, "c..CrossValidator.q_range=[%d, %d]" % (st, ed)
            else:
                print >> out, line.strip()
    st = ed
    ed = st + nb_q - 1

print "done"
