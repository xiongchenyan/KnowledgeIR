"""
get all candidate entities from the spotter
"""

import sys
import json

from knowledge4ir.utils import term2lm
from knowledge4ir.utils import TARGET_TEXT_FIELDS


def get_per_ana_entities(line):
    h = json.loads(line)

    spot_data = h.get('spot', [])

    if type(spot_data) == dict:
        l_ana = []
        for field in TARGET_TEXT_FIELDS:
            l_ana.extend(spot_data[field])
    else:
        l_ana = spot_data

    l_e = sum([[ana[0] for ana in item[-1]] for item in l_ana], [])

    return l_e


def get_all_spotted_entities(in_name, out_name):
    l_total_e = []

    for line in open(in_name):
        l_e = get_per_ana_entities(line)
        l_total_e.extend(l_e)

    h_e_tf = term2lm(l_total_e)
    l_e_tf = h_e_tf.items()
    l_e_tf.sort(key=lambda  item: item[1], reverse=True)

    out = open(out_name, 'w')
    for e, cnt in l_e_tf:
        print >> out, e + '\t%d' % cnt
    out.close()
    return


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "I get all e id in data"
        print "2 para: q info or d info, spotted + out name"
        sys.exit(-1)
    get_all_spotted_entities(*sys.argv[1:])



