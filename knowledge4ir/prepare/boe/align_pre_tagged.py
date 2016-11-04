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
    for line in open(ana_in):
        docno = line.split()[0]
        if docno not in h_doc_info:
            continue
        ana_str = line.split('#')[-1].strip()
        cols = ana_str.split('\t')
        p = 0
        l_ana = []
        while p + 8 <= len(cols):
            st = cols[p + 2]
            ed = cols[p + 3]
            rho = cols[p + 4]
            name = cols[p + 7]
            fb_id = cols[p + 6]
            l_ana.append([fb_id, st, ed, {'score': rho}, name])
            p += 8
        cnt += 1
        h_doc_info[docno].update({'tagme': l_ana})
        print >> out, docno + '\t' + json.dumps(h_doc_info[docno])
        if not (cnt % 100):
            logging.info('aligned [%d] doc', cnt)
    out.close()
    logging.info('total aligned [%d/%d] doc', cnt, len(h_doc_info))
    return


def main(doc_info_in, ana_in, out_name):
    h_doc_info = load_doc_info(doc_info_in)
    align_ana(ana_in, h_doc_info, out_name)


if __name__ == '__main__':
    if 4 != len(sys.argv):
        print "I align tagme results for doc info"
        print "3 para: doc info in + ana in + out_name"
        sys.exit()

    main(*sys.argv[1:])




