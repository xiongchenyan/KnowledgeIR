"""
check how many input document's
    frequency, position, VS title entity
if treat title as salient entity (e.g. wiki and paper)
then check whether simple frequency and position (for wiki it is position)
will do all
"""

import json
from knowledge4ir.utils import term2lm
from knowledge4ir.utils import (
    body_field,
    title_field
)
from knowledge4ir.utils.boe import form_boe_per_field
import logging


def get_top_frequency(doc_info):
    l_ana = form_boe_per_field(doc_info, body_field)
    l_e = [ana['id'] for ana in l_ana]
    l_name = [ana['surface'] for ana in l_ana]
    h_e_name = dict(zip(l_e, l_name))
    h_e_tf = term2lm(l_e)
    l_e_tf = h_e_tf.items()
    l_e_tf.sort(key=lambda item: item[1], reverse=True)
    top_e_name = ""
    if l_e_tf:
        top_e_name = h_e_name[l_e_tf[0][0]]
    return l_e_tf, top_e_name


def check_title_e_rank(doc_info):
    """
    get the rank of title entity in frequency
    :param doc_info:
    :return:
    """
    l_e_tf, top_e_name = get_top_frequency(doc_info)
    h_e_rank = dict(zip([item[0] for item in l_e_tf], range(1, 1 + len(l_e_tf))))
    l_ana = form_boe_per_field(doc_info, title_field)
    if not l_ana:
        return None
    title_e = l_ana[0]['id']
    rank = h_e_rank.get(title_e, 0)
    if rank != 1:
        if l_e_tf:
            top_e, top_tf = l_e_tf[0]
            title_e, title_tf = l_e_tf[rank - 1]
            print doc_info[title_field] + '\t' + top_e_name + '\t%s\t%d\t%s\t%d' % (title_e, title_tf, top_e, top_tf)
    return rank


def title_e_freq_rank_distribution(in_name):
    """
    read doc in in_name
    get rank of title e in body's frequency count
    return the count of rank (and %)
    :param in_name: doc info with spot
    :return: l_dist[i] = # title e in rank i, 0 is missing, 101 is after 100
    """
    l_dist = [0] * 102
    for p, line in enumerate(open(in_name)):
        if not p % 100:
            logging.info("processed [%d] doc", p)
        h = json.loads(line)
        rank = check_title_e_rank(h)
        if rank is None:
            continue
        rank = min(101, rank)
        l_dist[rank] += 1
    return l_dist


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 3 != len(sys.argv):
        print "check title entity frequency rank in body"
        print "2 para: input + output"
        sys.exit(-1)
    l_dist = title_e_freq_rank_distribution(sys.argv[1])
    cnt = float(sum(l_dist))
    l_prob = [dist / cnt for dist in l_dist]
    h = {'title entity rank position count': l_dist,
         'title entity rank position probability': l_prob
         }
    json.dump(h, open(sys.argv[2], 'w'),  indent=1)
    logging.info('finished')


