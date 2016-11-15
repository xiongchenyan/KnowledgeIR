"""
perfect switch model
merge two ranking using ground truth, choosing the one that performs better
input:
    eval 1
    eval 2
output:
    max of eval 1,2
    relative improvement of mean NDCG
"""

from knowledge4ir.utils import load_gdeval_res
import sys
import random


def pick_best(l_qid_eva_a, l_qid_eva_b, prob=1.0):
    h_q_eva_b = dict(l_qid_eva_b)
    l_qid_best_eva = []
    mean_ndcg = 0
    mean_err = 0
    for qid, (ndcg, err) in l_qid_eva_a:
        ndcg_b, err_b = h_q_eva_b[qid]
        best_ndcg = max(ndcg, ndcg_b)
        best_err = max(err, err_b)
        worst_ndcg = min(ndcg, ndcg_b)
        worst_err = min(err, err_b)
        if random.random() <= prob:
            l_qid_best_eva.append([qid, (best_ndcg, best_err)])
        else:
            l_qid_best_eva.append([qid, (worst_ndcg, worst_err)])
        mean_ndcg += best_ndcg
        mean_err += best_err
    if l_qid_best_eva:
        mean_ndcg /= len(l_qid_best_eva)
        mean_err /= len(l_qid_best_eva)

    return l_qid_best_eva, mean_ndcg, mean_err


def perfect_merge(eva_a_in, eva_b_in):
    l_q_eva_a, ndcg_a, err_a = load_gdeval_res(eva_a_in)
    l_q_eva_b, ndcg_b, err_b = load_gdeval_res(eva_b_in)
    for p in xrange(11):
        prob = p * 0.1
        l_q_best_eva, best_ndcg, best_err = pick_best(l_q_eva_a, l_q_eva_b, prob)
        print "%.2f,amean,%.6f,%.6f" % (prob, best_ndcg, best_err)
        print '%.2f,relative,%.4f,%.4f' % (prob, best_ndcg / max(ndcg_a, ndcg_b) - 1,
                                           best_err / max(err_a, err_b) - 1)
    return


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "I do switch between the best eva results with prob"
        print "2 para: eva 1 + eva 2"
        sys.exit(-1)
    perfect_merge(*sys.argv[1:])




