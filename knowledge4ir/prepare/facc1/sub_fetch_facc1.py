"""
submit fetch_facc1.py
input:
    trec rank
    FACC1 dir
    out dir
output:
    one job for each file in FACC1 dir
"""

import os
from knowledge4ir.utils.condor import qsub_job
import sys
from knowledge4ir.utils import set_basic_log


set_basic_log()
if 4 != len(sys.argv):
    print "3 para: Trec rank + FACC1 dir + out dir"
    sys.exit(-1)

for dir_name, sub_dirs, file_names in os.walk(sys.argv[2]):
    for fname in file_names:
        in_name = os.path.join(dir_name, fname)
        out_name = in_name.replace(sys.argv[2], sys.argv[3])
        l_cmd = ['python', 'fetch_facc1.py', sys.argv[1], in_name, out_name]
        qsub_job(l_cmd)

print "all submitted"

