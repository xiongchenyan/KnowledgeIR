"""
input:
    fb rdf dump
output:
    a: [fb types] \t desp
    b: notable type \t desp

"""

import sys
from knowledge4ir.utils import (
    FbDumpReader,
    FbDumpParser,
)
import json

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def get_type_text(rdf_in, out_name):
    reader = FbDumpReader()
    parser = FbDumpParser()
    type_out = open(out_name + '.type', "w")
    notable_out = open(out_name + '.notable', 'w')
    t_cnt = 0
    n_cnt = 0
    for cnt, ll_col in enumerate(reader.read(rdf_in)):
        if not cnt % 0:
            print "processed [%d] obj, [%d] type text, [%d] notable text" % (cnt, t_cnt, n_cnt)
        oid = parser.get_obj_id(ll_col)
        if not oid:
            continue
        desp = parser.get_desp(ll_col)
        if not desp:
            continue
        l_type = parser.get_type(ll_col)
        notable_type = parser.get_notable(ll_col)
        desp = ' '.join(desp.split())
        if l_type:
            print >> type_out, json.dumps(l_type) + '\t' + desp
            t_cnt += 1
        if notable_type:
            print >> notable_out, json.dumps(notable_type) + '\t' + desp
            n_cnt += 1

    type_out.close()
    notable_out.close()

if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "I get type and notable texts from fb rdf dump"
        print "2 para: fb rdf + out pre"
        sys.exit(-1)
    get_type_text(*sys.argv[1:])







