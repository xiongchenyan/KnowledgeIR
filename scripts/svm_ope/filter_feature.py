"""
filter features
input:
    svm data
    feature dict, with only those to keep
output:
    filtered svm data
    new feature dict
"""

from knowledge4ir.utils import (
    load_svm_feature,
    dump_svm_feature
)
import json
import sys
from copy import deepcopy


def filter_feature(l_svm_data, h_feature_id):
    h_id_re_index = {}
    l_fid = h_feature_id.values()
    l_fid.sort()
    for fid in l_fid:
        h_id_re_index[fid] = len(h_id_re_index) + 1

    h_new_feature_id = dict([(name, h_id_re_index[fid])
                             for name, fid in h_feature_id.items()
                             ])

    l_new_svm = []
    for svm_data in l_svm_data:
        h_feature = dict([(h_id_re_index[fid], score)
                          for fid, score in svm_data['feature'].items()
                          if fid in h_id_re_index
                          ])
        new_svm_data = deepcopy(svm_data)
        new_svm_data['feature'] = h_feature
        l_new_svm.append(new_svm_data)
    return l_new_svm, h_new_feature_id


if __name__ == '__main__':

    if 4 != len(sys.argv):
        print "filter svm feature"
        print "3 para: svm data + feature name dict, names to keep only + out pre"
        sys.exit()

    h_feature_id = json.load(open(sys.argv[2]))
    l_svm_data = load_svm_feature(sys.argv[1])

    l_new_svm, h_new_feature_id = filter_feature(l_svm_data, h_feature_id)

    dump_svm_feature(l_new_svm, sys.argv[3])
    json.dump(h_new_feature_id, open(sys.argv[3] + '_feature_name', 'w'), indent=1)
    print "done"

