"""
split svm data to cw09 and cw12
input svm
output:
    data.cw09/cw12
"""

import sys
import shutil

from knowledge4ir.utils import (
    load_svm_feature,
    dump_svm_feature
)

if 2 != len(sys.argv):
    print "1 para: svm to split"
    sys.exit(-1)

l_data = load_svm_feature(sys.argv[1])
l_cw09 = [data for data in l_data if int(data['qid']) <= 200]
l_cw12 = [data for data in l_data if int(data['qid']) > 200]

dump_svm_feature(l_cw09, sys.argv[1] + '.cw09')
dump_svm_feature(l_cw12, sys.argv[1] + '.cw12')


shutil.copyfile(sys.argv[1] + '_name.json', sys.argv[1] + '.cw09_name.json')
shutil.copyfile(sys.argv[1] + '_name.json', sys.argv[1] + '.cw12_name.json')
