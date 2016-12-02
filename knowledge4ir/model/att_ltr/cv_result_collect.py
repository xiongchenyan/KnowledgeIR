"""
collect cv results
"""

import sys
import subprocess
import os
from knowledge4ir.utils import (
    GDEVAL_PATH,
)

if 3 != len(sys.argv):
    print "collect cv results"
    print '2 para: cv dir + qrel'
    sys.exit()

l_rank_lines = []

for dir_name, sub_dirs, file_names in os.walk(sys.argv[1]):
    for file_name in file_names:
        if file_name == 'trec':
            if 'Fold' in dir_name:
                l_rank_lines.extend(open(dir_name + '/' + file_name).read().splitlines())

rank_out_name = os.path.join(sys.argv[1], 'trec')
print >> open(rank_out_name, 'w'), '\n'.join(l_rank_lines).strip()

for d in [1, 3, 5, 10, 20]:
    eva_out = subprocess.check_output([
        'perl', GDEVAL_PATH, '-k', '%d' %d, sys.argv[2], rank_out_name])
    print >> open(os.path.join(sys.argv[1], 'eval.d%02d' % d), 'w'), eva_out.strip()
    print "d %d: %s" % (d, eva_out.splitlines()[-1])

print "finished"
