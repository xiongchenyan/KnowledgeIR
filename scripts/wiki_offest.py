"""
fix wiki offset
"""

import json

def fix_body_offset(doc_info):
    title_len = len(doc_info.get('title', "").split())
    if not 'spot' in doc_info:
        return
    if not 'bodyText' in doc_info['spot']:
        return
    l_ana = doc_info['spot']['bodyText']
    text = doc_info['bodyText']
    l_t = text.split()
    for i in xrange(len(l_ana)):
        l_ana[i]['loc'] = l_ana[i]['loc'][0] - title_len, l_ana[i]['loc'][1] - title_len
        new_sf = ' '.join(l_t[l_ana[i]['loc'][0]: l_ana[i]['loc'][1]])
        if new_sf != l_ana[i]['surface']:
            print '[%s] != [%s]' % (new_sf, l_ana[i]['surface'])
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
        print >> out, h
    print "finished"
    out.close()
