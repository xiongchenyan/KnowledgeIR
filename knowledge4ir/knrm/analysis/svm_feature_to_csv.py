"""
feature to csv
input:
    svm data
    feature name
output:
    csv, with head
"""

import json
from knowledge4ir.utils import load_svm_feature
import sys

if 4 != len(sys.argv):
    print "convert svm feature to csv format"
    print "3 para: svm in + feature name in + out"
    sys.exit(-1)

h_feature_name = json.load(open(sys.argv[2]))

l_feature_name = h_feature_name.items()
l_feature_name.sort(key=lambda item: item[1])
l_feature_name = [item[0] for item in l_feature_name]
head_str = 'qid,docno,label,' + ','.join(l_feature_name)

out = open(sys.argv[3], 'w')
print >> out, head_str

l_svm_data = load_svm_feature(sys.argv[1])
for svm_data in l_svm_data:
    line = svm_data['qid'] + ',' + svm_data['comment'] + ',%d,' + svm_data['score']
    h_feature = svm_data['feature']
    l_feature = h_feature.items()
    l_feature.sort(key=lambda item: item[0])
    l_feature_value = ['%.4f' % item[1] for item in l_feature]
    line += ','.join(l_feature_value)
    print >> out, line

out.close()
print "finished"
