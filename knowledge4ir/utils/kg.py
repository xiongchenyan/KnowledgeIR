"""
include kg reader
"""


import logging
import json


def read_kg(in_name):
    """
    read kg triples
    to h_object = {object id -> triple -> [tail, cnt] }
    :param in_name:
    :return:
    """
    h_kg = {}
    logging.info('start loading kg from [%s]', in_name)
    for line_cnt, line in enumerate(open(in_name)):
        cols = line.strip().split('\t')
        if len(cols) != 4:
            logging.warn('kg line [%s] error', line.strip())
            continue
        head, predicate, tail, cnt = cols
        tail = json.loads(tail)
        cnt = int(cnt)
        add_triple(h_kg, (head, predicate, tail, cnt))
        if 0 == (line_cnt % 10000):
            logging.info('load [%d] triples for [%d] entities',
                         line_cnt, len(h_kg))
    return h_kg


def add_triple(h_kg, triple):
    h, p, t, cnt = triple
    if h not in h_kg:
        h_kg[h] = {}
    if p not in h_kg[h]:
        h_kg[h][p] = []
    h_kg[h][p].append((t, cnt))


def merge_triple_cnt(h_kg):
    h_new = {}
    logging.info('start merging triple counts')
    for h, h_predicate in h_kg.items():
        h_new[h] = {}
        for p, l_tails in h_predicate.items():
            h_new_tail = {}
            for t, cnt in l_tails:
                if t not in h_new_tail:
                    h_new_tail[t] = cnt
                else:
                    h_new_tail[t] += cnt
            h_new[h][p] = h_new_tail.items()
            logging.info('[%s-%s] from [%d -> %d]', h, p, len(l_tails), len(h_new_tail))

    logging.info('merged')
    return h_new


def fetch_tail(h_kg, head, predicate):
    if head not in h_kg:
        return None
    if predicate not in h_kg[head]:
        return None
    return h_kg[head][predicate]


def dump_kg(h_kg, out_name):
    logging.info('start dumping...')
    l_head_triple = sorted(h_kg.items(),
                           key=lambda item: (item[0].split('/')[1], int(item[0].split('/')[-1]))
                           )
    out = open(out_name, 'w')
    triple_cnt = 0
    for head, h_predicate in l_head_triple:
        for predicate, l_trails in h_predicate.items():
            for tail, df in l_trails:
                print >> out, '%s\t%s\t%s\t%d' % (head, predicate, json.dumps(tail), df)
        triple_cnt += len(h_predicate)
    out.close()
    logging.info('dumped to [%d] triples for [%d] entities',
                 triple_cnt, len(l_head_triple))
    return


def read_surface_form(kg_in):
    """
    load surface form from kg
    all names will be lower cased
    :return:
    """
    h_name_cnt = {}
    for line_cnt, line in enumerate(open(kg_in)):
        line = line.strip()
        cols = line.split('\t')
        mid, predicate, tail, cnt = cols
        tail = json.loads(tail)
        cnt = int(cnt)
        if cnt < 3:
            continue
        if predicate not in {'/name', '/alias'}:
            continue
        tail = tail.lower()
        if tail not in h_name_cnt:
            h_name_cnt[tail] = (mid, cnt)
        else:
            if h_name_cnt[tail][1] < cnt:
                h_name_cnt[tail] = (mid, cnt)
    l_items = [(item[0], item[1][0]) for item in h_name_cnt.items()]
    h_surface_name = dict(l_items)
    logging.info('loaded [%d] entity surface names', len(h_surface_name))
    return h_surface_name


def load_nlss_dict(nlss_dump, max_nlss_per_e=100):
    """
    load nlss dump from entity_grid (from wiki)
    :param nlss_dump: input nlss data
    :param max_nlss_per_e: maximum number of nlss loaded per entity
    :return: h_nlss, e_id -> [(sent, l_e_id in sent)]
    """
    h_nlss = dict()

    logging.info('loading nlss from [%s]', nlss_dump)
    cnt = 0
    skipped_cnt = 0
    for p, line in enumerate(open(nlss_dump)):
        cnt += 1
        if not p % 10000:
            logging.info('loaded [%d] nlss line', p)
        cols = line.strip().split('\t')
        assert len(cols) >= 4
        e_id, s_id = cols[:2]
        sent = json.loads(cols[2])
        l_e = json.loads(cols[3])

        if e_id not in h_nlss:
            h_nlss[e_id] = []
        if len(h_nlss[e_id]) < max_nlss_per_e:
            h_nlss[e_id].append((sent, l_e))
        else:
            skipped_cnt += 1

    logging.info('loaded [%d] nlss for [%d] entities, max [%d] per e, skiped [%d] nlss',
                 cnt, len(h_nlss), max_nlss_per_e, skipped_cnt)
    return h_nlss

