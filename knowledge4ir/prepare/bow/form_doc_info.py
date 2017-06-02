"""
form doc  info
input:
    parse doc's raw text: docno \t title \t body text
output:
    json format doc infor, each line a json
"""

import json
import sys
reload(sys)
sys.setdefaultencoding('UTF8')


if 3 != len(sys.argv):
    print "i pack doc raw text to json format doc info"
    print "2 para: docno text in + output"
    sys.exit()

out = open(sys.argv[2], 'w')
err_cnt = 0
for line in open(sys.argv[1]):
    cols = line.strip().split('\t')
    if len(cols) < 3:
        err_cnt += 1
        continue
    docno, title = cols[:2]
    text = '\t'.join(cols[2:])
    h = dict()
    h['docno'] = docno
    h['title'] = title
    h['bodyText'] = text

    print >> out, json.dumps(h)

out.close()
print "done [%d] line error" % err_cnt
