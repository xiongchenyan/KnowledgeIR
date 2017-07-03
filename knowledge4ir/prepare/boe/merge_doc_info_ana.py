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
    title_field,
    load_json_info,
)
from itertools import izip
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def merge_boe(h_doc_info_base, h_doc_info_update):
    if 'tagme' in h_doc_info_base:
        h_doc_info_base['tagme'].update(h_doc_info_update['tagme'])
    else:
        h_doc_info_base['spot'].update(h_doc_info_update['spot'])
    return h_doc_info_base


def merge_raw_field(h_doc_info_base, h_doc_info_update):
    h_doc_info_base.update(h_doc_info_update)
    return h_doc_info_update


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
    out = open(out_name, 'w')
    with open(base_info_in) as base_in, open(update_info_in) as update_in:
        for line_base, line_update in izip(base_in, update_in):
            h_base_info = json.loads(line_base)
            h_update_info = json.loads(line_update)
            assert h_base_info['docno'] == h_update_info['docno']
            docno = h_base_info['docno']
            if merge_format == 'spot':
                h_total_info = merge_boe(h_base_info, h_update_info)
            else:
                h_total_info = merge_raw_field(h_base_info, h_update_info)

            print >> out, json.dumps(h_total_info)
            logging.info('[%s] merged', docno)

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
        print "3+ para: tagme info a + tagme info b + merged out name + update format: (spot|all)"
        print "update the a using b, if update using spot (default), then only spot field is updated"
        print "if to update all, all fields will be updated at the first level (no recurse)"
        print "make sure both files are ordered the same with docno 1-1 correspondence in each line"
        sys.exit(-1)
    tag_in_a, tag_in_b = sys.argv[1:3]
    out_name = sys.argv[3]
    merge_format = 'spot'
    if len(sys.argv) >= 4:
        merge_format = sys.argv[4]
    merge(tag_in_a, tag_in_b, out_name, merge_format)




