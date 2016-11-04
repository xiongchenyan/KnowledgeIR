"""
prepare textual fields for target entities
from dump
fields: name, alias, desp
in:
    dump rbf (filtered)
    target entities
out:
    json.dumps({'id':/m/1, 'name':asdv, 'alias':, "desp":})
"""

import json
import logging
from knowledge4ir.utils import (
    FbDumpParser,
    FbDumpReader,
)


def prepare_textual_fields(dump_in, target_in, out_name):
    s_target = set([line.split()[0] for line in open(target_in)])

    reader = FbDumpReader()

    parser = FbDumpParser()
    out = open(out_name, 'w')
    for cnt, l_v_col in enumerate(reader.read(dump_in)):
        mid = parser.get_obj_id(l_v_col)
        if 0 == (cnt % 10000):
            logging.info('processed %d obj', cnt)
        if mid not in s_target:
            continue
        desp = parser.get_desp(l_v_col)
        name = parser.get_name(l_v_col)
        alias = parser.get_alias(l_v_col)

        h = dict()
        h['id'] = mid
        h['desp'] = desp
        h['name'] = name
        h['alias'] = alias

        print >> out, json.dumps(h)

    out.close()

    logging.info('finished')


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log

    set_basic_log(logging.INFO)

    if 4 != len(sys.argv):
        print "I construct json format entity textual fields"
        print "3 para: rbf in + target entities in (col [0] is entity) + out name"
        sys.exit()

    prepare_textual_fields(*sys.argv[1:])







