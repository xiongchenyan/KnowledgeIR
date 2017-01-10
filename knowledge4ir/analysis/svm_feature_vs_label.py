"""
the histogram of relevant pairs vs irrelevant ones
"""

from knowledge4ir.utils import (
    load_svm_feature,
    add_svm_feature,
)
import json
import math


def get_feature_avg(svm_in):
    l_svm_data = load_svm_feature(svm_in)
    h_rel_feature = {}
    h_irrel_feature = {}
    rel_cnt = 0
    irrel_cnt = 0

    for data in l_svm_data:
        score = data['score']
        h_feature = data['feature']
        if score > 0:
            h_rel_feature = add_svm_feature(h_rel_feature, h_feature)
            rel_cnt += 1
        else:
            h_irrel_feature = add_svm_feature(h_irrel_feature, h_feature)
            irrel_cnt += 1

    rel_cnt = float(rel_cnt)
    irrel_cnt = float(irrel_cnt)
    for key in h_rel_feature:
        h_rel_feature[key] /= rel_cnt
    for key in h_irrel_feature:
        h_irrel_feature[key] /= irrel_cnt
    return h_rel_feature, h_irrel_feature


def process(svm_in, out_name):
    out = open(out_name, 'w')

    h_rel, h_irrel = get_feature_avg(svm_in)
    l_key_score = h_rel.items()
    l_key_score.sort(key=lambda item: item[0])

    print >> out, "relevance feature avg:"
    for key, score in l_key_score:
        print >> out, '%d\t%f' % (key, score)

    l_key_score = h_irrel.items()

    l_key_score.sort(key=lambda item: item[0])

    print >> out, 'irrelevance feature avg:'
    for key, score in l_key_score:
        print >> out, '%d\t%f' % (key, score)

    out.close()
    print "finished"


if __name__ == '__main__':
    import sys

    if 3 != len(sys.argv):
        print "calc feature average in relevant and other"
        print "2 para: svm in + out"
        sys.exit(-1)

    process(*sys.argv[1:])

