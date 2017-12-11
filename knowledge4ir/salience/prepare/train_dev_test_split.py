"""
split train dev and test via given docno
input:
    whole json corpus with docno field
    train docno's one per line
    dev docno's one per line
    test docno's one per line
output:
    train docs
    dev docs
    test docs
"""

import sys
import json


if 6 != len(sys.argv):
    print "split to train test and dev via given docno in test and dev"
    print "5 para: json corpus + train doc no + dev doc no + test doc no + out pre"
    sys.exit(-1)


s_train_docno = set(open(sys.argv[2]).read().splitlines())
s_dev_docno = set(open(sys.argv[3]).read().splitlines())
s_test_docno = set(open(sys.argv[4]).read().splitlines())
print '[%d] dev [%d] test doc nos' % (len(s_dev_docno), len(s_test_docno))
train_out = open(sys.argv[5] + '.train', 'w')
dev_out = open(sys.argv[5] + '.dev', 'w')
test_out = open(sys.argv[5] + '.test', 'w')

train_cnt, dev_cnt, test_cnt = 0, 0, 0

for p, line in enumerate(open(sys.argv[1])):
    line = line.strip()
    docno = json.loads(line)['docno']
    if docno in s_dev_docno:
        print >> dev_out, line
        dev_cnt += 1
    elif docno in s_test_docno:
        print >> test_out, line
        test_cnt += 1
    elif docno in s_train_docno:
        print >> train_out, line
        train_cnt += 1

train_out.close()
dev_out.close()
test_out.close()
print "finished [%d][%d][%d] train/dev/test" % (train_cnt, dev_cnt, test_cnt)






