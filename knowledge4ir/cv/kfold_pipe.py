"""
run kfold cv pipeline on given svm data
input:
    svm data
do:
    partition the data
    run cross validation
    evaluate
output:
    cv_res
"""

from scholarranking.letor.kfold_cv_run import RanklibRunner
from scholarranking.letor.kfold_partition import kfold_svm_data
import sys
import os
import logging
from scholarranking.utils import set_basic_log, load_py_config

q_st = 1
q_ed = 100

set_basic_log(logging.INFO)

if 4 != len(sys.argv):
    print "kfold pipe"
    print "3 para: svm feature in + model conf + result out dir"
    print "config file:"
    RanklibRunner.class_print_help()
    sys.exit()

svm_in = sys.argv[1]
out_dir = sys.argv[3]

kfold_dir = os.path.join(out_dir, 'kfolds')
cv_dir = os.path.join(out_dir, 'cv_dir')

if not os.path.exists(kfold_dir):
    os.makedirs(kfold_dir)
if not os.path.exists(cv_dir):
    os.makedirs(cv_dir)

conf = load_py_config(sys.argv[2])
runner = RanklibRunner(config=conf)
with_dev = runner.with_dev
kfold_svm_data(svm_in, q_st, q_ed, kfold_dir, conf.RanklibRunner.nb_fold, with_dev)
runner.cross_validation(kfold_dir, cv_dir)


