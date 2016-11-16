"""
switch by query entity linking accuracy
input:
    the one using entity's eval
    the baseline eval
    q info with corresponding ($1's) query annotation
    q linking
output:
    relative improvements (over the best individual method) using different entity accuracy threshold
"""

from knowledge4ir.utils import load_gdeval_res, load_query_info
import json

l_tagger = ['tagme', 'cmns']


def calc_q_link_accuracy(q_info_in, q_manual_info_in):
    h_qid_info = load_query_info(q_info_in)
    h_qid_manual_info = load_query_info(q_manual_info_in)
    h_q_f1 = {}
    for qid, h_info in h_qid_info.items():
        l_e = []
        if 'tagme' in h_info:
            l_e = [ana[0] for ana in h_info['tagme']['query']]
        elif 'cmns' in h_info:
            l_e = [ana[0] for ana in h_info['cmns']['query']]

        l_label_e = [ana[0] for ana in h_qid_manual_info[qid]['manual']['query']]
        if len(l_e) == 0 & len(l_label_e) == 0:
            h_q_f1[qid] = 1
            continue
        s_e = set(l_e)
        s_true = set(l_label_e)
        prec = 0
        recall = 0
        overlap = float(len(s_e.intersection(s_true)))
        if s_e:
            prec = overlap / len(s_e)
        if s_true:
            recall = overlap / len(s_true)
        if prec == 0 | recall == 0:
            f1 = 0
        else:
            f1 = 2.0 * prec * recall / ( prec + recall )
        h_q_f1[qid] = f1

    return h_q_f1


def pick_via_q_linking_accuracy(l_qid_eva_a, l_qid_eva_b, h_q_f1, f1_bar=1.0):
    h_q_eva_b = dict(l_qid_eva_b)
    l_qid_best_eva = []
    mean_ndcg = 0
    mean_err = 0
    for qid, (ndcg, err) in l_qid_eva_a:
        ndcg_b, err_b = h_q_eva_b[qid]
        f1 = h_q_f1[qid]
        if f1 <= f1_bar:
            l_qid_best_eva.append([qid, (ndcg_b, err_b)])
        else:
            l_qid_best_eva.append([qid, (ndcg, err)])

    if l_qid_best_eva:
        mean_ndcg /= sum([item[1][0] for item in l_qid_best_eva]) / float(len(l_qid_best_eva))
        mean_err /= sum([item[1][1] for item in l_qid_best_eva]) / loat(len(l_qid_best_eva))

    return l_qid_best_eva, mean_ndcg, mean_err


def linking_merge(eva_a_in, eva_b_in, q_info_in, q_manual_info_in):
    l_qid_eva_a, ndcg_a, err_a = load_gdeval_res(eva_a_in)
    l_qid_eva_b, ndcg_b, err_b = load_gdeval_res(eva_b_in)
    h_q_f1 = calc_q_link_accuracy(q_info_in, q_manual_info_in)
    for p in xrange(11):
        f1_bar = p * 0.1
        l_q_merge_eva, merge_ndcg, merge_err = pick_via_q_linking_accuracy(
            l_qid_eva_a, l_qid_eva_b, h_q_f1, f1_bar)
        # print "%.2f,amean,%.6f,%.6f" % (prob, best_ndcg, best_err)
        print '%.2f%%,relative,' % (f1_bar * 100) + \
              "{0:.02f}%".format((merge_ndcg / max(ndcg_a, ndcg_b) - 1) * 100) + "," + \
              "{0:.02f}%".format((merge_err / max(err_a, err_b) - 1) * 100)
    return


if __name__ == '__main__':
    import sys
    if 5 != len(sys.argv):
        print "I do switch based on q entity linking accuracy"
        print "4 para: eva 1 + eva 2 + q info + q manual"
        sys.exit(-1)
    linking_merge(*sys.argv[1:])



