"""
tagme tagged results to doc info
input:
    tagme format (plus tagged column)
        docno \t title \t body \t # tagged (title or body text)
    wiki id -> fb id dict
output:
    doc info in json format
        doc no \t json dict


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


def wrap_doc(line, h_wiki_fb, tagged_field):
    cols = line.split('#')
    doc_str = '#'.join(cols[:-1])
    tagged_str = cols[-1]

    docno, title, body = doc_str.split('\t')[:3]

    l_tagged = tagged_str.split('\t')
    p = 0
    l_ana = []
    while p + 6 <= len(l_tagged):
        wiki_id, surface, st, ed, score, name = l_tagged[p: p + 6]
        if wiki_id not in h_wiki_fb:
            logging.warn('[%s] not in wiki-> fb dict', wiki_id)
            p += 6
            continue
        obj_id = h_wiki_fb[wiki_id]
        try:
            ana = [obj_id, int(st), int(ed), {'score': float(score)}, name]
        except ValueError:
            logging.warn('ana %s format error', json.dumps(l_tagged[p: p + 6]))
            p += 6
            continue
        l_ana.append(ana)
        p += 6

    h = dict()
    h[title_field] = title
    h[body_field] = body_field
    h['docno'] = docno
    h['tagme'] = dict()
    h['tagme'][tagged_field] = l_ana
    return docno, h


def process(tagme_in, wiki_fb_dict_in, out_name, tagged_field):
    h_wiki_fb = dict([line.strip().split('\t')[:2] for line in open(wiki_fb_dict_in)])
    logging.info('wiki fb dict loaded')
    out = open(out_name, 'w')

    for cnt, line in enumerate(open(tagme_in)):
        if not cnt % 1000:
            logging.info('process [%d] lines', cnt)
        docno, h = wrap_doc(line.strip(), h_wiki_fb, tagged_field)
        print >> out, docno + '\t' + json.dumps(h)

    out.close()
    logging.info('finished')

if __name__ == '__main__':
    if 5 != len(sys.argv):
        print "4 para: tag me out + wiki fb matching dict + out  + tagged field (title|bodyText)"
        sys.exit(-1)

    process(*sys.argv[1:])





