"""
tokenize and clean text
"""

import sys
from knowledge4ir.utils import raw_clean, set_basic_log
import logging

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

set_basic_log(logging.INFO)

if 3 != len(sys.argv):
    print "2 para: input + raw cleaned (tokenized, punctuation rm, and lower) output"
    sys.exit(-1)

out = open(sys.argv[2], 'w')
err_cnt = 0
for p, line in enumerate(open(sys.argv[1])):
    if not p % 1000:
        logging.info('processed [%d] line [%d] error', p, err_cnt)
    try:
        print >> out, raw_clean(line.strip())
    except UnicodeDecodeError:
        logging.warn('[%d] line decode error', p)
        err_cnt += 1

print "finished"
out.close()
