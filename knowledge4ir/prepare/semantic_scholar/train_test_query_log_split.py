"""
split the query log as train test
input:
    all log, in json format
    testing queries, in json format
output:
    split the log into train and test splits
    make sure all testing queries are in the test splits:
        if raw_clean query in testing, then consider it as a testing log
"""

import json
from knowledge4ir.utils import raw_clean

def _query_in_set(session_info, h_target_q):
    q = session_info['query']
    q = raw_clean(q)
    return q in h_target_q

def load_test_q(test_q_info_in):
    lines = open(test_q_info_in).read().splitlines()
    l_h = [json.loads(line) for line in lines]
    l_q_qid = [(h['query'], h['qid']) for h in l_h]
    return dict(l_q_qid)


def split_train_test(session_in, test_q_info_in, out_pre):
    h_test_q = load_test_q(test_q_info_in)
    train_out = open(out_pre + '_train.json', 'w')
    test_out = open(out_pre + '_test.json', 'w')
    train_cnt = 0
    test_cnt = 0
    for p, line in enumerate(open(session_in)):
        if not p % 100:
            print "processed [%d] session" % p
        h_session = json.loads(line)
        if _query_in_set(h_session, h_test_q):
            print >> test_out, line.strip()
            test_cnt += 1
        else:
            print >> train_out, line.strip()
            train_cnt += 1

    print "finished, [%d] train [%d] test" % (train_cnt, test_cnt)
    train_out.close()
    test_out.close()

if __name__ == '__main__':
    import sys
    if 4 != len(sys.argv):
        print "split training query log"
        print "3 para: all log in + test q info in + out pre"
        sys.exit(-1)
    split_train_test(*sys.argv[1:])




