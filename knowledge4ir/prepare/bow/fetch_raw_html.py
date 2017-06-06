"""
fetch target html for given docno's
use warc package
    for ClueWeb 12, the open source warc will do
    for ClueWeb 09, it is a little triky, will need the modified one

input:
    target doc no, one per line
    folder of raw warc format clueweb doc
do:
    read warc file, dump out html for target docno

output:
    a file, each line:
        docnot \t html
"""

import warccw09
import warc
import os
import ntpath
import logging
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

cw_version = '09'

def get_content(record, cw_v):
    if cw_v == '09':
        res = record.payload
        res = ' '.join(res.split())
        return res.strip()
    if cw_v == '12':
        res = ""
        for line in record.payload:
            res += line.strip() + ' '
        res = ' '.join(res.split())
        res = ' '.join(res.splitlines())
        return res.strip()
    return ""


def get_target_doc_per_file(fname, s_docno):
    global cw_version
    s_doc_pre = set(['-'.join(docno.split('-')[1:3]) for docno in s_docno])

    l_res = []
    cw09_pre = ntpath.basename(ntpath.dirname(fname)) + '-' + ntpath.basename(fname.replace('.warc.gz', ""))
    cw12_pre = ntpath.basename(fname).replace('.warc.gz', "")
    if (cw09_pre not in s_doc_pre) & (cw12_pre not in s_doc_pre):
        return l_res
    if cw_version == '09':
        in_file = warccw09.open(fname)
    else:
        in_file = warc.open(fname)
    logging.info('start reading [%s]', fname)
    cnt = 0
    try:
        for record in in_file:
            if 'warc-trec-id' not in record:
                logging.warn('record has no trec id')
                continue
            cnt += 1
            docno = record['warc-trec-id']
            logging.debug('get doc [%s]', docno)
            if docno not in s_docno:
                continue
            logging.info('get [%s]', docno)
            res = get_content(record, cw_version)
            l_res.append(docno + '\t' + res)
    except AssertionError:
        logging.error('[%s] assertion error', fname)
    logging.info('[%s] get [%d] target docs in [%d] doc', fname, len(l_res), cnt)
    return l_res


def process_dir(in_dir, target_doc_in, out_name, this_cw_version):
    global cw_version
    cw_version = this_cw_version
    s_docno = set(open(target_doc_in).read().splitlines())
    logging.info('total [%d] target docno', len(s_docno))
    out = open(out_name, 'w')
    total_cnt = 0
    for dir_name, sub_dirs, file_names in os.walk(in_dir):
        for fname in file_names:
            in_name = os.path.join(dir_name, fname)
            l_res = get_target_doc_per_file(in_name, s_docno)
            if l_res:
                print >> out, '\n'.join(l_res)
                total_cnt += len(l_res)
    out.close()
    logging.info('finished, get [%d/%d] target docs', total_cnt, len(s_docno))


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log(logging.INFO)

    if 5 != len(sys.argv):
        print "4 para: in_dir + target docno + output + cw version(09|12)"
        sys.exit(-1)

    process_dir(*sys.argv[1:])







