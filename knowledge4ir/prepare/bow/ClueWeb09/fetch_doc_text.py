"""
fetch the texts for docno's in trec ranking file
    for now there is no fields
input:
    trec ranking
    doc_text file each file is a partition parsed text
        docno \t url \t text
output:
    doc info file
        docno \t h{'docno:', 'field(now it is bodyText and fake title)':text}

"""

import sys
import json
import logging
from knowledge4ir.utils import load_trec_ranking_with_score
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def fetch_doc_text(trec_rank_in, doc_text_in, out_name):
    l_q_ranking = load_trec_ranking_with_score(trec_rank_in)
    ll_docno = [[docno for docno, __ in rank] for __, rank in l_q_ranking]
    s_target_docno = set(sum(ll_docno, []))
    logging.info('[%d] target docno', len(s_target_docno))
    err_cnt = 0
    cnt = 0
    out = open(out_name, 'w')
    for line in open(doc_text_in):
        cols = line.strip().split('\t')
        if len(cols) != 3:
            logging.warning('text format error %s', json.dumps(cols))
            err_cnt += 1
            continue
        docno, url, text = cols
        if docno in s_target_docno:
            logging.info('find [%s]', docno)
            h = dict()
            h['docno'] = docno
            h['bodyText'] = text
            h['title'] = ' '.join(text.split()[:10])
            print >>out, docno + '\t' + json.dumps(h)
            cnt += 1
    out.close()
    logging.info('finished [%s], found [%d], err [%d]', cnt, err_cnt)


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 4 != len(sys.argv):
        print "I fetch doc text for ClueWeb"
        print "3 para: trec ranking + doc text file + out name"
        sys.exit()

    fetch_doc_text(*sys.argv[1:])




