"""
input:
annotated doc info
output:
    word-entity joint corpus
    one line of text
    one line of text with e id replacing surface form
"""

import json
import sys
l_field = ['title', 'bodyText']
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def replace_surface(text, l_ana):
    l_t = text.lower().split()
    l_res = []
    current_p = 0
    for ana in l_ana:
        st, ed = ana['loc']
        e_id = ana['entities'][0]['id']
        l_res.extend(l_t[current_p:st])
        l_res.append(e_id)
        current_p = ed
    return ' '.join(l_res)


def convert_per_doc(h_d_info):
    l_converted_line = []
    for field in l_field:
        if field not in h_d_info:
            continue
        text = h_d_info[field]
        l_ana = h_d_info['spot'][field]
        l_converted_line.append(replace_surface(text, l_ana))
    return l_converted_line


def process(in_name, out_name):
    out = open(out_name, 'w')
    for p, line in enumerate(open(in_name)):
        if not p % 10000:
            print "processed [%d] lines" % p
        h = json.loads(line)
        print >> out, '\n'.join(convert_per_doc(h))

    out.close()
    print "finished"


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "2 para: doc info to get joint corpus + output"
        sys.exit(-1)
    process(*sys.argv[1:])
