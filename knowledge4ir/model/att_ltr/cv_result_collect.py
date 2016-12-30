"""
collect cv results
"""

import sys
import subprocess
import os
from knowledge4ir.utils import (
    GDEVAL_PATH,
    QREL_IN,
)


def collect_cv_results(cv_dir, qrel_in):
    l_rank_lines = []
    cnt = 0
    for dir_name, sub_dirs, file_names in os.walk(cv_dir):
        for file_name in file_names:
            if file_name == 'trec':
                if 'Fold' in dir_name:
                    cnt += 1
                    l_rank_lines.extend(open(dir_name + '/' + file_name).read().splitlines())
    rank_out_name = os.path.join(sys.argv[1], 'trec')
    print >> open(rank_out_name, 'w'), '\n'.join(l_rank_lines).strip()

    for d in [1, 3, 5, 10, 20]:
        eva_out = subprocess.check_output([
            'perl', GDEVAL_PATH, '-k', '%d' %d, qrel_in, rank_out_name])
        print >> open(os.path.join(sys.argv[1], 'eval.d%02d' % d), 'w'), eva_out.strip()
        print "d %d: %s" % (d, ','.join(eva_out.splitlines()[-1].split(',')))
    print "finished with [%d] fold" % cnt


if __name__ == '__main__':

    if 2 > len(sys.argv):
        print "collect cv results"
        print '1 para: cv dir + qrel (opt, default %s)' % QREL_IN
        sys.exit()
    qrel_in = QREL_IN
    if len(sys.argv) > 2:
        qrel_in = sys.argv[2]
    cv_dir = sys.argv[1]
    collect_cv_results(cv_dir, qrel_in)
