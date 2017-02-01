"""
json sparql result to trec rank
"""

import json
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def url_transfer(uri):
    res = '<dbpedia:' + uri.split('/')[-1] + '>'
    return res


def q_transfer(q, pre):
    return 'QALD2_' + pre + '-' + q


def convert_per_line(line, pre):
    cols = line.strip().split('\t')
    qid = q_transfer(cols[0], pre)
    h = json.loads('\t'.join(cols[1:]))

    l_bindings = h.get('results', {}).get('bindings', {})

    l_uri = []
    for binding in l_bindings:
        uri = binding.get('uri', {}).get('value', "")
        uri = url_transfer(uri)
        l_uri.append(uri)

    l_res = [qid + ' Q0 ' + uri + ' 1' for uri in l_uri]
    return l_res


def process(in_name, out_name):
    pre = 'tr'
    if 'test' in in_name:
        pre = 'te'

    out = open(out_name, 'w')
    for line in open(in_name):
        l_res = convert_per_line(line, pre)
        if l_res:
            print >> out, "\n".join(l_res)
    out.close()
    print "done"

if __name__ == '__main__':

    if 3 != len(sys.argv):
        print "2 para: sparql res + out"
        sys.exit(-1)

    process(*sys.argv[1:])







