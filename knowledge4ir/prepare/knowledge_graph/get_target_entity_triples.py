"""
get target entity triples
    mainly query entities
input:
    fb rbf dump
output:
    a file of e -> l_vcol [(entity, predicate, tail)] list


"""

from knowledge4ir.utils import (
    FbDumpReader,
    FbDumpParser,
    set_basic_log,
)
import sys
import logging
import json
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


set_basic_log()
if 4 != len(sys.argv):
    print "3 para: fb rbf dump + target entity + out "
    sys.exit(-1)


s_entity = set([line.strip().split()[0] for line in open(sys.argv[2])])

reader = FbDumpReader()
cnt = 0
parser = FbDumpParser()
h_res = {}
out = open(sys.argv[3],'w')
for o_cnt, l_v_col in enumerate(reader.read(sys.argv[1])):
    if not o_cnt % 1000:
        logging.info('[%d] record [%d] target obj', o_cnt, cnt)
    oid = parser.get_obj_id(l_v_col)
    if not oid:
        continue
    if oid not in s_entity:
        continue
    # h_res[oid] = l_v_col
    h = dict()
    h['id'] = oid

    l_edges = []
    for v_col in l_v_col:
        edge = parser.discard_prefix(v_col[1])
        tail = v_col[2]
        possible_id = parser.get_id_for_col(tail)
        if possible_id:
            tail = possible_id
        else:
            tail = parser.discard_prefix(tail)
        l_edges.append([edge, tail])
    h['edges'] = l_edges
    print >> out, json.dumps(h)
    cnt += 1

logging.info('total [%d/%d] entity triples get', cnt, len(s_entity))
# json.dump(h_res, open(sys.argv[3], 'w'), indent=1)

logging.info('finished')


