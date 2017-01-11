"""
get top k entities in doc, from results of rank_component_analysis
"""

import sys
import json


def get_top_k(in_name, out_name):
    h_q_top_rel = {}
    h_q_top_irrel = {}

    for line in open(in_name):
        cols = line.strip().split('\t')
        q, doc, label = cols[:3]
        if q not in h_q_top_rel:
            h_q_top_rel[q] = {}
            h_q_top_irrel[q] = {}
        l_top_k = json.loads(cols[-1])
        l_top_k = l_top_k[:3]
        label = int(label)
        if label > 0:
            add_top_k(h_q_top_rel[q], l_top_k)
        else:
            add_top_k(h_q_top_irrel[q], l_top_k)

    out = open(out_name, 'w')
    for q in h_q_top_rel.keys():
        l_rel = h_q_top_rel[q].items()
        l_irrel = h_q_top_irrel.get(q, {}).items()
        l_rel.sort(key=lambda item: -item[1])
        l_irrel.sort(key=lambda item: -item[1])
        print >> out, '%s\trel\t%s' % (q, json.dumps(l_rel))
        print >> out, '%s\tirl\t%s' % (q, json.dumps(l_irrel))

    out.close()
    print "done"


def add_top_k(h, l_top_k):
    for e, name, __ in l_top_k:
        if (e, name) not in h:
            h[(e, name)] = 1
        else:
            h[(e, name)] += 1

    return


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "2 para: esearch.json + out"
        sys.exit()

    get_top_k(*sys.argv[1:])




