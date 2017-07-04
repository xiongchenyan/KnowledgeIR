"""
check the stats of grid, per doc, and whole corpus
stats include:
    # of sent per doc (with entity)
    # of entity per doc
    as the sent->entity grid:
        in degree of entity
        out degree of sent
    only in body field
"""

import json
from knowledge4ir.utils import body_field
import logging


def count_per_doc(h_doc_info):
    l_sent_grid = h_doc_info['e_grid'].get(body_field, [])
    l_sent_cnt = []
    h_entity_cnt = {}
    for sent_grid in l_sent_grid:
        l_ana = sent_grid.get('spot', [])
        l_sent_cnt.append(len(l_ana))
        for ana in l_ana:
            e_id = ana['id']
            h_entity_cnt[e_id] = h_entity_cnt.get(e_id, 0) + 1
    return l_sent_cnt, h_entity_cnt


def count_grid_stats(doc_info_in, out_name):
    nb_sent = 0
    nb_in_degree = 0
    nb_out_degree = 0
    nb_e = 0
    cnt = 0
    out = open(out_name, 'w')
    for line in open(doc_info_in):
        if not cnt % 1000:
            logging.info('counting [%d] line', cnt)
        cnt += 1
        h = json.loads(line)
        l_sent_cnt, h_entity_cnt = count_per_doc(h)
        this_nb_sent = len(l_sent_cnt)
        this_nb_e = len(h_entity_cnt)
        this_nb_out_degree, this_nb_in_degree = 0, 0
        if this_nb_sent:
            this_nb_out_degree = sum(l_sent_cnt) / float(this_nb_sent)
        if this_nb_e:
            this_nb_in_degree = sum([item[1] for item in h_entity_cnt.items()]) / float(this_nb_e)
        h_res = {'docno': h['docno']}
        h_res['nb_sent'] = this_nb_sent
        h_res['nb_e'] = this_nb_e
        h_res['nb_out_degree'] = this_nb_out_degree
        h_res['nb_in_degree'] = this_nb_in_degree
        print >> out, json.dumps(h_res)
        nb_sent += this_nb_sent
        nb_in_degree += this_nb_in_degree
        nb_out_degree += this_nb_out_degree
        nb_e += this_nb_e
    cnt = float(cnt)
    h_res = {'docno': 'total'}
    h_res['nb_sent'] = nb_sent / cnt
    h_res['nb_e'] = nb_e / cnt
    h_res['nb_out_degree'] = nb_out_degree / cnt
    h_res['nb_in_degree'] = nb_in_degree / cnt

    json.dump(h_res, open(out_name + '.meta', 'w'), indent=1)
    logging.info('finished')
    return

if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 3 != len(sys.argv):
        print "count grid simple stats"
        print "2 para: doc info with e_grid + out_name"
        sys.exit(-1)
    count_grid_stats(*sys.argv[1:])




