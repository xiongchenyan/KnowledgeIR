"""
prepare kg edge from freebase for embedding
entity vocabulary:
    all those has wiki link
kg edge:
    three:
        desp's terms
        all neighbor entities
        type

input:
    fb rbf dump
output:
    3 files each one a entity \t tail pair
    desp
    neighbor
    type
"""

from knowledge4ir.utils import (
    FbDumpReader,
    FbDumpParser,
    set_basic_log,
)
import sys
import logging


set_basic_log()
if 3 != len(sys.argv):
    print "2 para: fb rbf dump + out pre"
    sys.exit(-1)


reader = FbDumpReader()
cnt = 0
desp_out = open(sys.argv[2] + '_desp', 'w')
type_out = open(sys.argv[2] + '_type', 'w')
neighbor_out = open(sys.argv[2] + '_neighbor', 'w')
parser = FbDumpParser()
for o_cnt, l_v_col in enumerate(reader.read(sys.argv[1])):
    if not o_cnt % 1000:
        logging.info('[%d] record [%d] wiki obj', o_cnt, cnt)
    oid = parser.get_obj_id(l_v_col)
    if not oid:
        continue
    wid = parser.get_wiki_id(l_v_col)
    if not wid:
        continue

    desp = parser.get_desp(l_v_col)
    l_type = parser.get_type(l_v_col)
    l_neighbor = parser.get_neighbor(l_v_col)
    cnt += 1
    for term in desp.lower().split():
        print >> desp_out, oid + ' ' + term
    for type_s in l_type:
        print >> type_out, oid + ' ' + type_s
    for __, n_e in l_neighbor:
        print >> neighbor_out, oid + ' ' + n_e

desp_out.close()
type_out.close()
neighbor_out.close()
logging.info('finished')



