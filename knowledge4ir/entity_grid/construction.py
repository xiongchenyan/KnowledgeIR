"""
construct the entity grid
sentence -> e id

input:
    spotted documents
    with texts, and annotations
output:
    add the entity_grid field to the dict:
    entity_grid -> fields
        -> list of sentences -> ana with location altered to the sentence's offset

    can directly dump pretty print format (with each line one sentence)
        docno \t sentence: list of ana in each line
"""

from knowledge4ir.utils.boe import form_boe_per_field
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    QUERY_FIELD,
)
import json
import sys
from nltk.tokenize import sent_tokenize
import logging
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def construct_per_text(text, l_ana):
    """

    :param text:
    :param l_ana: list of ana in form_boe_per_field's format
    :return: l_e_grid = [ {'sent':, 'spot':}  ]
    """

    l_sent = sent_tokenize(text)
    ll_ana = _align_ana_to_sent(l_sent, l_ana)
    l_e_grid = []
    for sent, l_ana in zip(l_sent, ll_ana):
        h_sent = dict()
        h_sent['sent'] = sent
        h_sent['spot'] = l_ana
        l_e_grid.append(h_sent)

    return l_e_grid


def _align_ana_to_sent(l_sent, l_ana):
    """
    track the token pos in l_sent,
    put ana's into corresponding sentences, and modify the offset to per sent's
    :param l_sent: sentences in the text
    :param l_ana: annotations in the text, ana={'surface': sf, 'loc': loc, 'id': e}
    :return:
    """
    ll_per_sent_ana = []
    sent_st, sent_ed = 0, 0
    ana_p = 0
    l_ana.sort(key=lambda item: item['loc'][0])
    for sent in l_sent:
        l_t = sent.split()
        sent_ed = sent_st + len(l_t)
        logging.info('sent [%s] at [%d:%d), ana_p at %s', sent, sent_st, sent_ed,
                     json.dumps(l_ana[ana_p]['loc']))
        l_this_ana = []
        while ana_p < len(l_ana):
            ana = l_ana[ana_p]
            st, ed = ana['loc']
            if (st >= sent_st) & (ed <= sent_ed):
                st -= sent_st
                ed -= sent_st
                ana['loc'] = (st, ed)
                l_this_ana.append(ana)
                ana_p += 1
                if ana['surface'] != ' '.join(l_t[st:ed]):
                    logging.warn('[%s] != [%s], in [%s] [%d:%d)',
                                 ana['surface'], ' '.join(l_t[st:ed]),
                                 sent, st, ed)
            elif st < sent_st:
                ana_p += 1
            else:
                break
        sent_st = sent_ed
        ll_per_sent_ana.append(l_this_ana)

    logging.info('[%d] ana matched [%d]', len(l_ana), ana_p)
    return ll_per_sent_ana


def construct_per_doc(doc_info, l_target_field):
    doc_info['e_grid'] = {}
    for field in l_target_field:
        l_ana = form_boe_per_field(doc_info, field)
        text = doc_info.get(field, "")
        l_e_grid = construct_per_text(text, l_ana)
        doc_info['e_grid'][field] = l_e_grid
    return doc_info


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 3 > len(sys.argv):
        print "form entity grid"
        print "2+ para: tagged doc in + out_name + out format:full (default)|pretty"
        sys.exit(-1)

    out = open(sys.argv[2], 'w')
    out_format = 'full'
    if len(sys.argv) > 3:
        out_format = sys.argv[3]

    for p, line in enumerate(open(sys.argv[1])):
        if not p % 100:
            logging.info('processed [%d] doc', p)
        doc_info = json.loads(line)
        doc_info = construct_per_doc(doc_info, TARGET_TEXT_FIELDS)
        if out_format == 'full':
            print >> out, json.dumps(doc_info)
        else:
            docno = doc_info.get('docno', "")
            if not docno:
                docno = doc_info.get('title', 'NA')
            for field in TARGET_TEXT_FIELDS:
                l_e_grid = doc_info['e_grid'][field]
                for h_grid in l_e_grid:
                    print >> out, docno + '\t' + field + '\t' + json.dumps(h_grid)

    out.close()
    logging.info('finished')


