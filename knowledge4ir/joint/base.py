"""
"""

import logging
import json


def load_doc_info(doc_info_in):
    h_doc_info = {}
    logging.info('start loading doc info')

    for p, line in enumerate(open(doc_info_in)):
        h = json.loads(line)
        h_doc_info[h['docno']] = h
        if not p % 1000:
            logging.info('loaded [%d] doc', p)
    return h_doc_info



