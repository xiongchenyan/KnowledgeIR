"""
make svm with different bins
input:
    svm with bin features
    feature_name
output:
    one svm for each amount of bins
    0, 0-1, 0-2, 0-nb_bin

"""

from knowledge4ir.feature.svm_ope.filter_feature import filter_feature
import json
from knowledge4ir.utils import (
    load_svm_feature,
    dump_svm_feature,
)


def make_one_bin(l_svm_data, h_total_feature_id, out_name, max_bin):
    h_new_feature_id = {}
    for name, fid in h_total_feature_id.items():
        if 'Maxbin' not in name:
            h_new_feature_id[name] = fid
            continue

        bin_n = int(name.split('_')[-1])
        if bin_n <= max_bin:
            h_new_feature_id[name] = fid

    l_new_svm, h_feature_id = filter_feature(l_svm_data, h_new_feature_id)
    dump_svm_feature(l_new_svm, out_name)
    json.dump(h_feature_id, open(out_name + '_feature_name', 'w'), indent=1)


def main(svm_in, feature_name_in, out_pre, nb_bin=5):
    l_svm_data = load_svm_feature(svm_in)
    h_total_feature_id = json.load(open(feature_name_in))

    for max_bin in xrange(nb_bin - 1):
        make_one_bin(l_svm_data, h_total_feature_id, out_pre + 'First_%02d' % max_bin, max_bin)
        print 'first bin %d done' % max_bin
    print 'done'


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "bin filter"
        print "svm in + out"
        sys.exit()
    svm_in = sys.argv[1]
    out_pre = sys.argv[2]
    feature_in = sys.argv[1] + '_feature_name'
    main(svm_in, feature_in, out_pre)

