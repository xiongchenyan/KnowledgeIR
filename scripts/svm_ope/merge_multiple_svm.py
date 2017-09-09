"""
merge multiple svm data
add features one by one
input:
    list of svm data
output:
    one merged
"""


from knowledge4ir.utils import (
    load_svm_feature,
    dump_svm_feature,
)
import json


def load_multiple_svm_and_feature(svm_files_in):
    l_name_fields = open(svm_files_in).read().splitlines()
    l_names = [line.split('\t')[0] for line in l_name_fields]
    # l_fields = [line.split('\t')[1] for line in l_name_fields]
    ll_svm_data = [load_svm_feature(name) for name in l_names]
    l_h_feature_name = [json.load(open(name + '_name.json')) for name in l_names]
    l_new_h_feature_name = []
    for h_feature_name in l_h_feature_name:
        l_new_h_feature_name.append(h_feature_name)
    return ll_svm_data, l_new_h_feature_name


def form_feature_mapping(l_h_feature_name):
    l_h_feature_id_name = [dict(zip(h.values(), h.keys())) for h in l_h_feature_name]
    h_universal_feature_name = {}
    for h in l_h_feature_name:
        for name in sorted(h.keys()):
            if name not in h_universal_feature_name:
                h_universal_feature_name[name] = len(h_universal_feature_name) + 1
    return h_universal_feature_name, l_h_feature_id_name


def align_per_q_d(l_h_features, l_h_feature_id_name, h_universal_feature_name):
    h_new_feature = {}
    for h_feature, h_feature_id_name in zip(l_h_features, l_h_feature_id_name):
        for fid, value in h_feature.items():
            new_id = h_universal_feature_name[h_feature_id_name[fid]]
            h_new_feature[new_id] = value
    return h_new_feature


def align_multiple_svm(ll_svm_data, l_h_feature_id_name, h_universal_feature_name):
    l_new_svm_data = []
    nb_data = len(ll_svm_data[0])
    for i in range(len(ll_svm_data)):
        assert nb_data == len(ll_svm_data[i])
        ll_svm_data[i].sort(key=lambda item: (int(item['qid']), item['comment']))

    for p in xrange(nb_data):
        l_this_pair_svm = [l_svm_data[p] for l_svm_data in ll_svm_data]
        qid = l_this_pair_svm[0]['qid']
        docno = l_this_pair_svm[0]['comment']
        for svm in l_this_pair_svm:
            assert qid == svm['qid']
            assert docno == svm['comment']
        qrel = l_this_pair_svm[0]['score']
        l_h_features = [svm['feature'] for svm in l_this_pair_svm]
        h_new_feature = align_per_q_d(l_h_features, l_h_feature_id_name, h_universal_feature_name)
        new_svm_data = {'qid': qid, 'score': qrel, 'comment': docno, 'feature': h_new_feature}
        l_new_svm_data.append(new_svm_data)

    return l_new_svm_data


def main(svm_files_in, out_name):
    ll_svm_data, l_h_feature_name = load_multiple_svm_and_feature(svm_files_in)
    print "loaded"
    h_universal_feature_name, l_h_feature_id_name = form_feature_mapping(l_h_feature_name)
    print "new feature names :%s" % (json.dumps(h_universal_feature_name, indent=1))
    l_new_svm_data = align_multiple_svm(ll_svm_data, l_h_feature_id_name, h_universal_feature_name)
    print "merged"
    dump_svm_feature(l_new_svm_data, out_name)
    json.dump(h_universal_feature_name, open(out_name + '_name.json', 'w'), indent=1)
    print "done"


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "merge multiple svms"
        print "2 para: svm file names in + out_name"
        sys.exit()
    main(*sys.argv[1:])



