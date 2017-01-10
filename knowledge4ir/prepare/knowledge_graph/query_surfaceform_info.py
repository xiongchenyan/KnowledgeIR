"""
get the surface form infor for a query
including:
    surface form cnt
    surface form -> top 5 candidate cnt
input:
    q_info.tagme, to get target surface form
    FACC1 SurfacrForm dict
output:
    surface form: top 5 e, cnt
"""

from knowledge4ir.utils import (
    load_query_info,
)
import json
import math


def get_target_surfaceforms(q_info_in):
    h_qid_info = load_query_info(q_info_in)
    h_surface = dict()
    for qid, h_info in h_qid_info.items():
        query = h_info['query']
        for ana in h_info['tagme']['query']:
            sf = query[ana[1]:ana[2]]
            h_surface[sf.lower()] = []
    print "total [%d] sf" % (len(h_surface))
    return h_surface


def get_top_k_candidate(surface_form_dict_in, h_surface, k):
    target_cnt = 0
    for p, line in enumerate(open(surface_form_dict_in)):
        sf, eid, cnt = line.strip().split('\t')
        cnt = math.floor(float(cnt))
        if sf not in h_surface:
            continue
        target_cnt += 1
        if len(h_surface[sf]) < k:
            h_surface[sf].append((eid, cnt))
            h_surface[sf].sort(key=lambda item: item[1])
        else:
            min_id, min_cnt = h_surface[sf][0]
            if cnt < min_cnt:
                h_surface[sf][0] = (eid, cnt)
                h_surface[sf].append((min_id, cnt))
            else:
                h_surface[sf].append((eid, cnt))

        if not p % 1000:
            print "read [%d] sf line get [%d] target" % (p, target_cnt)

    h_sf_top_k = {}
    for sf, l in h_surface.items():
        l.sort(key=lambda item: -item[1])
        h_sf_top_k[sf] = l[:k]
    return h_sf_top_k


def process(q_info_in, sf_in, out_name):
    h_sf = get_target_surfaceforms(q_info_in)
    h_sf_topk = get_top_k_candidate(sf_in, h_sf, 5)
    json.dump(h_sf_topk, open(out_name, 'w'), indent=1)
    print "done"


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        print "I get target surface form's top candidates"
        print "3 para: q_info in  + surface dict + out"
        sys.exit(-1)

    process(*sys.argv[1:])



