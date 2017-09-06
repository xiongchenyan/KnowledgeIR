"""
check
    the fraction of abstract entities appear in the body or title
    the fraction of #1 frequent body entity in abstract
    the fraction of first position entity (title|body) in abstract
input:
    hashed nyt
output:
    each line:
        docno, abs entity cnt, fraction of abs e in title, fraction of abs e in body
        whether #1 body entity is salient, whether #1 positioned entity is salient
"""

import json
from knowledge4ir.utils import term2lm


def process_one_doc(h_doc):
    h_spot = h_doc.get('spot', {})
    l_abs_e = h_spot.get('abstract', [])
    l_body_e = h_spot.get('bodyText', [])
    l_title_e = h_spot.get('title', [])
    docno = h_doc['docno']
    s_a_e = set(l_abs_e)
    s_b_e = set(l_body_e)
    s_t_e = set(l_title_e)

    nb_abs_e = len(set(l_abs_e))
    nb_abs_e_in_body = len([e for e in set(l_abs_e) if e in s_b_e])
    nb_abs_e_in_title = len([e for e in set(l_abs_e) if e in s_t_e])

    first_e_salient = int(l_body_e[0] in s_a_e)
    h_b_e = term2lm(l_body_e)
    l_b_e_tf = h_b_e.items()
    l_b_e_tf.sort(key=lambda item: -item[1])
    freq_e_salience = int(l_b_e_tf[0][0] in s_a_e)

    return docno, nb_abs_e, nb_abs_e_in_title, nb_abs_e_in_body, freq_e_salience, first_e_salient


def process(hashed_doc_in, out_name):
    out = open(out_name, "w")
    l_head = ['doc', 'nb abs e', 'nb abs e in title',
              'nb abs e in body', 'most frequent e in abs', 'first e in abs']
    print >> out, ','.join(l_head)
    nb_abs_e = 0
    l_total = [0, 0, 0, 0, 0]

    for p, line in enumerate(open(hashed_doc_in)):
        if not p % 1000:
            print "counted [%d] doc" % p
        l_data = process_one_doc(json.loads(line))
        print >> out, ','.join(['%d' % n for n in l_data])
        l_total = [l_total[p] + l_data[p + 1] for p in range(len(l_data))]

    print >> out, "TOTAL," + ','.join(['%d' % n for n in l_total])
    print "done"


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "count salience stat from hashed annotated corpus"
        print "2 para: hashed ana doc in + out"
        sys.exit(-1)
    process(*sys.argv[1:])


