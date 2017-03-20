"""
run one fold
in para:
    config
    data in
    out dir
    fold k
    with dev (0|1)
"""

from knowledge4ir.model.cross_validator import CrossValidator
import json
import sys
from knowledge4ir.utils import (
    load_py_config,
)

if len(sys.argv) < 5:
    print "run one fold of cv"
    print "4+ para: config in + data in + out dir + fold k + with dev (optional, default 0, 0|1)"
    print "config:"
    CrossValidator.class_print_help()
    sys.exit(-1)

with_dev = 0
if len(sys.argv) > 6:
    with_dev = int(sys.argv[5])

conf = load_py_config(sys.argv[1])
in_name = sys.argv[2]
out_dir = sys.argv[3]
fold_k = int(sys.argv[4])
cv = CrossValidator(config=conf)
if with_dev:
    cv.train_dev_test_one_fold(in_name, out_dir, fold_k)
else:
    cv.train_test_one_fold(in_name, out_dir, fold_k)


