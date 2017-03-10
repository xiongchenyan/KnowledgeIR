"""
prepare wiki surface form stats
input:
    pre-calculated Wiki surface form info file
output:
    sf dict of stats
        tf, lp, nbEntity, cmns_entropy
"""


import json
import sys
from scipy import stats


def convert_per_record(line):
    l = json.loads(line)
    sf, tf, linked_cnt, lp, h_entity = l
    nb_entity = len(h_entity)
    l_cmns = [item[1][1] for item in h_entity.items()]
    cmns_entropy = stats.entropy(l_cmns)

    h_stat = dict([('tf', tf), ('lp', lp), ('nb_entity', nb_entity), ('cmns_entropy', cmns_entropy)])
    return sf, h_stat


def wiki_sf_stat_prepare(in_name, out_name):
    h_res = {}
    for p, line in enumerate(open(in_name)):
        sf, h_stat = convert_per_record(line)
        h_res[sf] = h_stat
        if not p % 1000:
            print "converted [%d] lines" % p
    json.dump(h_res, open(out_name, 'w'), indent=1)
    return


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "wiki sf stat prepare:"
        print "2 para: wiki surface json input + output"
        sys.exit(-1)

    wiki_sf_stat_prepare(*sys.argv[1:])





