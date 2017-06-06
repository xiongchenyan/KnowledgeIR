"""
read and align the parsed result from MSRA
input:
    one partition of CW09 url - docno
    one partition of CW09 tokenized
output:
    docno \t url \t doctext
"""

import sys
import logging
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def align_doc_url(doc_text_in, doc_url_in, out_name):
    h_url_no = {}
    err_cnt = 0
    for line_cnt, line in enumerate(open(doc_url_in)):
        cols = line.strip().split('\t')
        if len(cols) != 2:
            err_cnt += 1
        url, docno = '\t'.join(cols[:-1]), cols[-1]
        h_url_no[url] = docno
        if not (line_cnt % 1000):
            logging.info('load [%d] url doc mapping', line_cnt)
    logging.info('%d doc url in this partition %d err url', len(h_url_no), err_cnt)
    out = open(out_name, "w")
    cnt = 0
    err_cnt = 0
    for line_cnt, line in enumerate(open(doc_text_in)):
        line = line.strip()
        cols = line.split('\t')
        if len(cols) != 3:
            logging.warning(line)
            err_cnt += 1

        url, text = '\t'.join(cols[:-2]), cols[-1]
        if url in h_url_no:
            docno = h_url_no[url]
            print >> out, docno + "\t" + url.replace('\t', '') + '\t' + text
            cnt += 1
        if not ( line_cnt % 10000):
            logging.info('read [%d] doc text', line_cnt)
    out.close()
    logging.info("finished [%s][%s] with [%d] found, [%d] text err",
                 doc_text_in, doc_url_in, cnt, err_cnt)


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if len(sys.argv) != 4:
        print "3 para: doc text in, doc url in, out_name"
        sys.exit()

    align_doc_url(*sys.argv[1:])

