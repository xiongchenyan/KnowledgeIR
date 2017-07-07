"""
for the info in folder, dump e id and words + e id
"""

from knowledge4ir.utils.boe import form_boe_per_field
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    QUERY_FIELD,
)
import os
import json


def dump_per_doc(in_name, out_name):
    e_out = open(out_name + '.entity', 'w')
    v_out = open(out_name + '.vocab', 'w')
    l_e = []
    l_v = []
    for line in open(in_name):
        h = json.loads(line)
        for field in [QUERY_FIELD] + TARGET_TEXT_FIELDS:
            l_ana = form_boe_per_field(h, field)
            text = h.get(field, "").lower()
            l_e.extend([ana['id'] for ana in l_ana])
            l_v.extend(text.split())

    l_e = list(set(l_e))
    print >> e_out, '\n'.join(l_e)
    l_v = list(set(l_v))
    print >> v_out, '\n'.join(l_e + l_v)
    e_out.close()
    v_out.close()
    print '[%s] vocab to [%s.entity + .vocab]' % (in_name, out_name)
    return


def process(in_dir, out_dir):
    print "get vocab from [%s] dir to [%s] folder" % (in_dir, out_dir)
    for dir_name, sub_dirs, l_file_names in os.walk(in_dir):
        for f_name in l_file_names:
            in_name = os.path.join(dir_name, f_name)
            out_name = os.path.join(out_dir, f_name)
            dump_per_doc(in_name, out_name)
    print "all finished"

if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "get vocabulary"
        print "splited info dir + out dir"
        sys.exit(-1)
    process(*sys.argv[1:])
