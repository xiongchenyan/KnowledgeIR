"""
condor run
"""

import sys
from knowledge4ir.utils.condor import qsub_job
import os
import json
import ntpath

if 4 != len(sys.argv):
    print "I submit fetch_doc_text.py"
    print "3 para: trec rank in + doc text dir + out dir"
    sys.exit()

for dir_name, sub_dirs, f_names in os.walk(sys.argv[2]):
    for f_name in f_names:
        doc_text_in = os.path.join(dir_name, f_name)
        out_name = doc_text_in.replace(dir_name, sys.argv[3])
        out_dir = ntpath.dirname(out_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        l_cmd = ['python', 'fetch_doc_text.py', sys.argv[1], doc_text_in, out_name]
        qsub_job(l_cmd)

print "done"

