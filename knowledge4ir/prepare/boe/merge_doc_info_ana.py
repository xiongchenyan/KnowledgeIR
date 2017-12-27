"""
merge doc info
input:
    doc info with title annotated by tagme
    doc info with body annotated by tagme
output:
    merge the two together
"""

import logging
import sys
import json
from knowledge4ir.utils import (
    body_field,
    paper_abstract_field,
    title_field,
    load_json_info,
    SPOT_FIELD,
)
from itertools import izip
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def get_docno(doc_info):
    if 'docno' in doc_info:
        docno = doc_info['docno']
    else:
        docno = doc_info['id']
    return docno


def merge_via_key_chain(h_doc_info_base, h_doc_info_update, l_key_chain):
    """
    update the h_doc_info_base by h_doc_info_update, only update the key_chain pointed field
    :param h_doc_info_base:
    :param h_doc_info_update:
    :param l_key_chain:
    :return:
    """

    h_base = h_doc_info_base
    h_update = h_doc_info_update
    for key in l_key_chain[:-1]:
        h_base = h_doc_info_base.get(key, {})
        h_update = h_doc_info_update.get(key, {})

    h_base[l_key_chain[-1]] = h_update.get(l_key_chain[-1], {})
    return h_doc_info_base

def _parse_key_chain(in_str):
    return in_str.split("#")


def merge_boe(h_doc_info_base, h_doc_info_update):
    if 'tagme' in h_doc_info_base:
        h_doc_info_base['tagme'].update(h_doc_info_update['tagme'])
    else:
        h_doc_info_base['spot'].update(h_doc_info_update['spot'])
    return h_doc_info_base


def merge_raw_field(h_doc_info_base, h_doc_info_update):
    # h_doc_info_base.update(h_doc_info_update)
    for field in h_doc_info_update.keys():
        if (field != 'tagme') & (field != 'spot'):
            h_doc_info_base[field] = h_doc_info_update[field]
    h_doc_info_base = merge_boe(h_doc_info_base, h_doc_info_update)
    return h_doc_info_base


def s2_replace(h_info):
    if body_field in h_info:
        h_info[paper_abstract_field] = h_info[body_field]
        del h_info[body_field]
    if SPOT_FIELD in h_info:
        if body_field in h_info[SPOT_FIELD]:
            h_info[SPOT_FIELD][paper_abstract_field] = h_info[SPOT_FIELD][body_field]
            del h_info[SPOT_FIELD][body_field]
    return h_info


def load_all_doc_info(title_info_in):
    logging.info('start loading doc info...')
    h_doc_h_info = {}
    for line in open(title_info_in):
        h_info = json.loads(line)
        docno = h_info['docno']
        h_doc_h_info[docno] = h_info
    logging.info('loaded [%d] docs')
    return h_doc_h_info


def merge(base_info_in, update_info_in, out_name, merge_format):
    """
    only the tagme annotation is merged
    :param base_info_in:
    :param update_info_in:
    :return:
    """
    l_key_chain = []
    if "#" in merge_format:
        l_key_chain = _parse_key_chain(merge_format)
    out = open(out_name, 'w')
    with open(base_info_in) as base_in, open(update_info_in) as update_in:
        cnt = 0
        for line_base, line_update in izip(base_in, update_in):
            if not cnt % 1000:
                logging.info('processed [%d] lines', cnt)
            cnt += 1
            h_base_info = json.loads(line_base)
            h_update_info = json.loads(line_update)
            base_docno = get_docno(h_base_info)
            update_docno = get_docno(h_update_info)
            assert base_docno == update_docno
            docno = base_docno
            if merge_format == 'spot':
                h_total_info = merge_boe(h_base_info, h_update_info)
            elif merge_format == 'all':
                h_total_info = merge_raw_field(h_base_info, h_update_info)
                # h_total_info = merge_boe(h_total_info, h_base_info)
            elif merge_format == 's2':
                h_base_info = s2_replace(h_base_info)
                h_update_info = s2_replace(h_update_info)
                h_total_info = merge_raw_field(h_base_info, h_update_info)
            elif l_key_chain:
                h_total_info = merge_via_key_chain(h_base_info, h_update_info, l_key_chain)
            else:
                raise NotImplementedError
            print >> out, json.dumps(h_total_info)
            logging.debug('[%s] merged', docno)

    # h_doc_h_info_base = load_json_info(base_info_in)
    # for line in open(update_info_in):
    #     h_update_info = json.loads(line)
    #     docno = h_update_info['docno']
    #     h_base_info = h_doc_h_info_base[docno]
    #     if merge_format == 'spot':
    #         h_total_info = merge_boe(h_base_info, h_update_info)
    #     else:
    #         h_total_info = merge_raw_field(h_base_info, h_update_info)
    #     print >> out, json.dumps(h_total_info)
    #     logging.info('[%s] merged', docno)

    logging.info('finished')
    out.close()


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 4 > len(sys.argv):
        print "3+ para: tagme info a + tagme info b + merged out name + update format: (spot,all,s2, or key chain)"
        print "update the a using b, if update using spot (default), then only spot field is updated"
        print "if to update all, all fields will be updated at the first level (no recurse)"
        print "make sure both files are ordered the same with docno 1-1 correspondence in each line"
        print "key chain is the list of keys, separated by #"
        sys.exit(-1)
    tag_in_a, tag_in_b = sys.argv[1:3]
    out_name = sys.argv[3]
    merge_format = 'spot'
    if len(sys.argv) >= 4:
        merge_format = sys.argv[4]
    merge(tag_in_a, tag_in_b, out_name, merge_format)




