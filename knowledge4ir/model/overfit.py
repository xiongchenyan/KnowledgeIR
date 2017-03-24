"""
overfit a data
"""

from knowledge4ir.model.cross_validator import CrossValidator
import sys
import logging
from knowledge4ir.utils import (
    load_py_config,
    set_basic_log
)


set_basic_log(logging.INFO)
if len(sys.argv) < 4:
    print "overfit data"
    print "3 para: config in + data in + out dir"
    print "config:"
    CrossValidator.class_print_help()
    sys.exit(-1)

conf = load_py_config(sys.argv[1])
in_name = sys.argv[2]
out_dir = sys.argv[3]
cv = CrossValidator(config=conf)
s_qid = set(['%d' % qid for qid in range(cv.q_range[0], cv.q_range[1] + 1)])
cv.train_test_generator(in_name, in_name, out_dir, s_qid, s_qid)


