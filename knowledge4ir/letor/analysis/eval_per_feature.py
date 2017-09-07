"""
evaluate the performance with a single feature
in:
    svm
    qrel
do:
    for each feature:
        form a trec ranking file
        call gdeval to evaluate
output:
    feature: ndcg, err

"""

from knowledge4ir.utils import (
    GDEVAL_PATH,
    dump_trec_ranking_with_score,
    seg_gdeval_out,
)
import subprocess
import json
import logging


def form_rank(svm_in, feature_d, w):
    h_q_ranking = {}

    for line in open(svm_in):
        cols = line.strip().split()
        qid = cols[1].split(':')[1]
        docno = cols[-1].strip()
        score = 0
        for col in cols:
            if col.startswith('%s:' % feature_d):
                score = float(col.split(':')[-1]) * w
                break
        if qid not in h_q_ranking:
            h_q_ranking[qid] = []
        h_q_ranking[qid].append([docno, score])
    logging.info('f [%d] ranking formed', feature_d)
    l_q_ranking = h_q_ranking.items()
    for i in xrange(len(l_q_ranking)):
        l_q_ranking[i][1].sort(key=lambda item: (-item[1], item[0]))
    return l_q_ranking


def eva_feature(svm_in, feature_d, out_pre, depth, w):
    l_q_ranking = form_rank(svm_in, feature_d, w)
    out_name = out_pre + '.tmp_trec_%d_%d' % (feature_d, int(w))
    dump_trec_ranking_with_score(l_q_ranking, out_name)
    eva_str = subprocess.check_output(['perl', GDEVAL_PATH, '-k', '%d' % depth, qrel_path, out_name])
    l_qid_eva, ndcg, err = seg_gdeval_out(eva_str, True)
    return l_qid_eva, ndcg, err


def main(svm_in, feature_name_in, out_name, depth):
    h_feature = json.load(open(feature_name_in))
    out = open(out_name, 'w')
    l_feature_d = h_feature.items()
    l_feature_d.sort(key=lambda item: item[0])
    for feature, d in l_feature_d:
        for name, w in zip(['', 'reverse_'], [1.0, -1.0]):
            __, ndcg, err = eva_feature(svm_in, d, out_name, depth, w)
            print >> out, '%s:%f,%f' % (name + feature, ndcg, err)
    out.close()
    logging.info('finished')


if __name__ == '__main__':
    import sys

    if 3 > len(sys.argv):
        print "svm in + qrel in + depth (opt default 20)"
        sys.exit(-1)
    global qrel_path
    qrel_path = sys.argv[2]
    d = 20
    if len(sys.argv) > 3:
        d = int(sys.argv[3])
    main(sys.argv[1], sys.argv[1] + '_name.json', sys.argv[1] + '.feature_eval', d)
