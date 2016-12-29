"""
read doc info
output two lines:
    1: body text
    2: body text with entity replaced by tagme's annotation
"""

from knowledge4ir.utils import body_field
import json
import sys


def replace(text, l_ana):
    e_text = ""
    p = 0
    for ana in l_ana:
        st, ed = ana[1], ana[2]
        e_text += text[p:st] + ana[0]
        p = ed

    return e_text


def process_doc(doc_info):
    text = doc_info[body_field]
    text = clean_text(text)
    l_ana = doc_info['tagme'][body_field]
    return text.lower(), replace(text, l_ana).lower()


def clean_text(text):
    l_t = text.split()
    s = set(['#'])
    l_new_t = [t for t in l_t if t not in s]
    return ' '.join(l_new_t)


def make_context_text(doc_info_in, out_name):
    out = open(out_name, 'w')
    for line in open(doc_info_in):
        d_info = json.loads('\t'.join(line.split('\t')[1:]))
        text, e_text = process_doc(d_info)
        print >> out, text
        print >> out, e_text

    out.close()
    print "done"


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "2 para: doc info + out"
        sys.exit(-1)

    make_context_text(*sys.argv[1:])



