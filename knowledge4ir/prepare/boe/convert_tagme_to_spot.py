"""
convert tagme format to spot format
so that the programs can run on both with no change
"""

import json
from knowledge4ir.utils import SPOT_FIELD


def spot_to_tagme(h_info):

    h_tagme_ana = h_info.get('tagme', {})
    h_spot_ana = {}
    for field, l_tagme_ana in h_tagme_ana.items():
        l_spot_ana = []
        for ana in l_tagme_ana:
            spot = {}
            e_id, st, ed, h_meta, sf = ana
            spot['surface'] = sf
            spot['loc'] = [st, ed]
            spot['entities'] = [{'id': e_id, 'cmns': h_meta['score']}]
            l_spot_ana.append(spot)
        h_spot_ana[field] = l_spot_ana

    del h_info['tagme']
    h_info[SPOT_FIELD] = h_spot_ana
    return h_info


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "convert tagme format to spot format"
        print "2 para: tagme info + spot info"
        sys.exit(-1)

    out = open(sys.argv[2], 'w')
    for p, line in enumerate(open(sys.argv[1])):
        if not p % 100:
            print "converted [%d] lines" % p
        h = json.loads(line)
        h = spot_to_tagme(h)
        print >> out, json.dumps(h)
    out.close()
    print "finished"
