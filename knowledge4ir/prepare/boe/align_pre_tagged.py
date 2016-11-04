"""
align pre tagged doc with doc info
input:
    doc info
    tagged docs
output:
    doc info with tagme
"""
import json
import sys
import logging
from knowledge4ir.utils import load_doc_info


def align_ana(ana_in, h_doc_info, out_name):
    out = open(out_name, 'w')
    cnt = 0
    s_dumped_docno = set()
    for line in open(ana_in):
        docno = line.split()[0]
        if docno not in h_doc_info:
            continue
        ana_str = line.split('#')[-1].strip()
        cols = ana_str.split('\t')
        p = 0
        l_ana = []
        while p + 8 <= len(cols):
            try:
                st = int(cols[p + 2])
                ed = int(cols[p + 3])
                rho = float(cols[p + 4])
                name = cols[p + 7]
                fb_id = cols[p + 6]
                l_ana.append([fb_id, st, ed, {'score': rho}, name])
            except ValueError:
                logging.warn('tagged cols mixed', json.dumps(cols[p: p + 8]))
            p += 8
        cnt += 1
        h = dict(h_doc_info[docno])
        h_tagme = {'bodyText': l_ana}
        title_len = 0
        if 'title' in h_doc_info[docno]:
            title_len = len(h_doc_info[docno]['title'])
        l_title_ana = [ana for ana in l_ana if ana[2] <= title_len]
        h_tagme['title'] = l_title_ana
        h['tagme'] = h_tagme

        print >> out, docno + '\t' + json.dumps(h)
        s_dumped_docno.add(docno)
        if not (cnt % 100):
            logging.info('aligned [%d] doc', cnt)
    logging.info('[%d/%d] has tag me ana', cnt, len(h_doc_info))
    for docno, h_info in h_doc_info.items():
        if docno not in s_dumped_docno:
            print >> out, docno + '\t' + json.dumps(h_doc_info[docno])
            cnt += 1
    out.close()
    logging.info('total dumped [%d/%d] doc', cnt, len(h_doc_info))
    return


def main(doc_info_in, ana_in, out_name):
    h_doc_info = load_doc_info(doc_info_in)
    align_ana(ana_in, h_doc_info, out_name)


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 4 != len(sys.argv):
        print "I align tagme results for doc info"
        print "3 para: doc info in + ana in + out_name"
        sys.exit()

    main(*sys.argv[1:])




