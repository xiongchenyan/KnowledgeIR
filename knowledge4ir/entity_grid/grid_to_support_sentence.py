"""
dump support sentences from entity grid
input:
    doc info with e_grid
        (wiki's for most usage)
    whether only keep with title entities (True|False)
output:
    e_id \t sent_id (docno_pos) \t sentence \t all entities
    (only keep one key for now, to get pairs, just do an intersection)
will not order via key, need another sort
"""

import json
from knowledge4ir.utils import (
    title_field,
    body_field,
    set_basic_log,
    load_json_info,
    E_GRID_FIELD,
    SPOT_FIELD,
)
from knowledge4ir.utils.boe import (
    form_boe_per_field,
)
import logging


def parse_doc_to_nlss(doc_info):
    l_nlss = []  # e_id, sent id, sent, l_e
    docno = doc_info.get('docno', "")
    if not docno:
        doc_info = doc_info.get(title_field)
    e_grid = doc_info.get(E_GRID_FIELD, {})
    if type(e_grid) is not dict:
        logging.FATAL('%s is not dict from [%s]', json.dumps(e_grid), docno)
        raise TypeError
    for p, sent_grid in enumerate(doc_info.get(E_GRID_FIELD, {}).get(body_field, [])):
        sent_id = docno + '_s%d' % p
        sent = sent_grid['sent']
        l_ana = sent_grid[SPOT_FIELD]
        l_e = list(set([ana['id'] for ana in l_ana]))
        for e in l_e:
            l_nlss.append([e, sent_id, sent, l_e])
    logging.debug('[%s] [%d] nlss pair', docno, len(l_nlss))
    return l_nlss


def filter_to_title_entity(l_nlss, doc_info):
    l_title_ana = form_boe_per_field(doc_info, title_field)
    s_e = set([ana['id'] for ana in l_title_ana])
    l_keep_nlss = []
    for nlss in l_nlss:
        l_e = nlss[-1]
        keep = False
        for e in l_e:
            if e in s_e:
                keep = True
                break
        if keep:
            l_keep_nlss.append(nlss)
    logging.debug('filter to title entity related only: [%d]->[%d]', len(l_nlss), len(l_keep_nlss))
    return l_keep_nlss


def dump_nlss(e_grid_in, out_name, restrict_to_title=False):
    """
    parse and dump nlss from e_grid
    :param e_grid_in: processed entity grids in json foramt
    :param out_name:
    :param restrict_to_title: whether to keep to title entities
    :return:
    """

    out = open(out_name, 'w')
    for p, line in enumerate(open(e_grid_in)):
        if not p % 1000:
            logging.info('parsed [%d] lines', p)
        doc_info = json.loads(line)
        l_nlss = parse_doc_to_nlss(doc_info)
        if restrict_to_title:
            l_nlss = filter_to_title_entity(l_nlss, doc_info)
        for key, sent_id, sent, l_e in l_nlss:
            print >> out, '\t'.join([key, sent_id, sent, json.dumps(l_e)])

    out.close()
    logging.info('finished')
    return

if __name__ == '__main__':
    import sys
    set_basic_log()
    if 3 > len(sys.argv):
        print "2+ para: e grid in + nlss out + restrict to title e or not (0, default|1)"
        sys.exit(-1)

    restrict_to_title = False
    if len(sys.argv) > 3:
        restrict_to_title = bool(int(sys.argv[3]))

    dump_nlss(sys.argv[1], sys.argv[2], restrict_to_title)




