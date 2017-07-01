"""
fix wiki offset
"""

import json
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    body_field,
)
from knowledge4ir.utils.boe import SPOT_FIELD

def fix_body_offset(doc_info):
    title_len = len(doc_info.get('title', "").split())
    if not SPOT_FIELD in doc_info:
        return
    if not body_field in doc_info['spot']:
        return
    l_ana = doc_info[SPOT_FIELD][body_field]
    text = doc_info[body_field]
    l_t = text.split()
    for i in xrange(1, len(l_ana)):
        loc = l_ana[i]['loc']
        loc[0] -= title_len
        loc[1] -= title_len
        l_ana[i]['loc'] = loc
        new_sf = ' '.join(l_t[loc[0]: loc[1]])
        if new_sf != l_ana[i]['surface']:
            print 'title len [%d]' % title_len
            print '[%s] != [%s]' % (new_sf, l_ana[i]['surface'])
    doc_info[SPOT_FIELD][body_field] = l_ana[1:]
    return

if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "re align offset"
        print "2 para: wiki in + out"
        sys.exit(-1)

    out = open(sys.argv[2], 'w')
    for p, line in enumerate(open(sys.argv[1])):
        if not p % 100:
            print 'processed [%d]' % p
        h = json.loads(line)
        fix_body_offset(h)
        print >> out, json.dumps(h)
    print "finished"
    out.close()
