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
)

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def merge_one(h_doc_info_title, h_doc_info_body):
    if 'tagme' in h_doc_info_title:
        h_doc_info_title['tagme'][body_field] = h_doc_info_body['tagme'][body_field]
    else:
        h_doc_info_title['spot'][body_field] = h_doc_info_body['spot'][body_field]
    return h_doc_info_title


def load_title_infos(title_info_in):
    logging.info('start loading title info...')
    h_doc_h_info = {}
    for line in open(title_info_in):
        h_info = json.loads(line)
        docno = h_info['docno']
        h_doc_h_info[docno] = h_info
    logging.info('loaded [%d] docs')
    return h_doc_h_info


def merge(title_info_in, body_info_in, out_name):
    """
    only the tagme annotation is merged
    :param title_info_in:
    :param body_info_in:
    :return:
    """

    h_doc_h_info_title = load_title_infos(title_info_in)
    out = open(out_name, 'w')
    for line in open(body_info_in):
        h_body_info = json.loads(line)
        docno = h_body_info['docno']
        h_title_info = h_doc_h_info_title[docno]
        h_total_info = merge_one(h_title_info, h_body_info)
        print >> out, json.dumps(h_total_info)
        logging.info('[%s] merged', docno)
    logging.info('finished')
    out.close()


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 4 != len(sys.argv):
        print "3 para: tagme info with title tagged + with body tagged + merged out name"
        sys.exit(-1)

    merge(*sys.argv[1:])




