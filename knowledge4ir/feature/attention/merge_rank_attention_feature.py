"""
merge rank and attention feature
input:
    rank feature
    attention feature
output:
    rank's ranking feature + attention feature
"""


import json
import sys
from copy import deepcopy

l_rank_name = ['qe_rank', 'qt_rank']
l_att_name = ['qe_att', 'qt_att']


def merge_meta(rank_name_in, att_name_in, out_name):
    h_r = json.load(open(rank_name_in))
    h_a = json.load(open(att_name_in))
    h = {}
    for rank_name in l_rank_name:
        h[rank_name] = h_r[rank_name]
    for att_name in l_att_name:
        h[att_name] = h_a[att_name]
    json.dump(h, open(out_name, 'w'))


def merge_one_pair(rank_info, att_info):
    merge_info = deepcopy(rank_info)
    merge_info['feature'][3:] = att_info['feature']
    return merge_info


def make_info_hash(in_name):
    h_info = {}
    for line in open(in_name):
        h = json.loads(line)
        key=h['q'] + '\t' + h['doc']
        h_info[key] = h
    return h_info


def merge(rank_in, att_in, out_name):
    for suf in ['_feature_name', '_feature_stat']:
        merge_meta(rank_in + suf, att_in + suf, out_name + suf)

    h_att_info = make_info_hash(att_in)
    out = open(out_name, 'w')
    for line in open(rank_in):
        h = json.loads(line)
        key = h['q'] + '\t' + h['doc']
        assert key in h_att_info
        merged_info = merge_one_pair(h, h_att_info[key])
        print >> out, json.dumps(merged_info)
    out.close()
    print "finished"

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "I merge att feature and ranking feature"
        print "3 para: ranking feature + att feature + out"
        sys.exit(-1)

    merge(*sys.argv[1:])


