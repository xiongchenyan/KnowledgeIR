"""
prepare surface form dict
input:
    surface form file:
        form \t fb id \t tf
    k:
        keep top k ( default 5)
output:
    a json dict
        form -> [fb id, tf]
"""


import sys
import json
import logging
from knowledge4ir.utils import (
    set_basic_log,
)

set_basic_log(logging.INFO)


if 3 > len(sys.argv):
    print "I prepare surface form dict in json format"
    print "2+ parameter: sf input + out name + top k (default 5)"
    sys.exit(-1)


h_sf = dict()
top_k = 5
if len(sys.argv) > 3:
    top_k = int(sys.argv[3])

for p, line in enumerate(open(sys.argv[1])):
    if not p % 1000:
        logging.info('processed [%d] lines, [%d] sf', p, len(h_sf))

    sf, fb_id, cnt = line.strip().split('\t')
    cnt = int(cnt)
    if sf not in h_sf:
        h_sf[sf] = [(fb_id, cnt)]
    else:
        h_sf[sf].append((fb_id, cnt))


logging.info('total [%d] sf get, calculating CMNS scores', len(h_sf))

h_sf_cmns = dict()

for sf, l_id_cnt in h_sf.items():
    l_id_cnt.sort(key=lambda item: -item[1])
    total_cnt = float(sum([item[1] for item in l_id_cnt]))

    l_top_candidate = [(item[0], item[1] / total_cnt) for item in l_id_cnt[:top_k]]
    h_sf_cmns[sf] = l_top_candidate

logging.info('dumping...')
json.dump(h_sf_cmns, open(sys.argv[2], 'w'), indent=1)
logging.info('finished, result dumped to [%s]', sys.argv[2])


