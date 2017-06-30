"""
rename entity to entities
"""


def re_name(h):
    if 'spot' not in h:
        return
    for field in h['spot']:
        l_ana = h['spot'].get(field, [])
        for i in xrange(len(l_ana)):
            new_ana = l_ana[i]
            new_ana['entities'] = new_ana.get('entity', [])
            del new_ana['entity']
            l_ana[i] = new_ana
        h['spot'][field] = l_ana
    return

if __name__ == '__main__':
    import json
    import sys
    if 3 != len(sys.argv):
        print "in json+ out name"
        sys.exit(-1)
    out = open(sys.argv[2], 'w')
    for line in open(sys.argv[1]):
        h = json.loads(line)
        re_name(h)
        print >> out, json.dumps(h)
    out.close()
    print "finished"