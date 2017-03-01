"""
partitation via query id
input:
    svm data
    qid range with qrel
output:
    sequential partition of qid in range to kfolds
    all else in another big testing file
    follow the same convention with letor 3.0
"""

from sklearn.cross_validation import KFold
import os
import json
import logging
default_K = 5


def kfold_q_pool(q_st, q_ed, nb_folds=default_K):
    l = range(q_st, q_ed + 1)
    h_test_fold = {}
    for p, (__, test_index) in enumerate(KFold(n=len(l), n_folds=nb_folds, shuffle=False)):
        for idx in test_index:
            h_test_fold['%d' % l[idx]] = p + 1
    print json.dumps(h_test_fold)
    return h_test_fold


def kfold_q_pool_uniform(q_st, q_ed, nb_folds=default_K):
    h_test_fold = {}
    for i in range(q_st, q_ed + 1):
        h_test_fold['%d' % i] = (i - q_st) % (nb_folds) + 1
    print json.dumps(h_test_fold)
    return h_test_fold


def kfold_svm_data(svm_in, q_st, q_ed, out_dir, nb_folds=default_K, with_dev=False):
    h_test_fold = kfold_q_pool_uniform(q_st, q_ed, nb_folds)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    l_test_out = []
    l_train_out = []
    if with_dev:
        l_dev_out = []
    else_out = open(os.path.join(out_dir, 'else.txt'), 'w')
    total_train_out = open(os.path.join(out_dir, 'total_train.txt'), 'w')
    for k in xrange(nb_folds):
        fold_dir = os.path.join(out_dir, 'Fold%d' % (k + 1))
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        test_name = os.path.join(fold_dir, 'test.txt')
        train_name = os.path.join(fold_dir, 'train.txt')
        if with_dev:
            dev_name = os.path.join(fold_dir, 'dev.txt')
            l_dev_out.append(open(dev_name, 'w'))
        l_test_out.append(open(test_name, 'w'))
        l_train_out.append(open(train_name, 'w'))

    logging.info('files created')

    for line in open(svm_in):
        line = line.strip()
        qid = line.split()[1].replace('qid:', '')

        if qid not in h_test_fold:
            # and else data
            print >> else_out, line
            continue
        print >> total_train_out, line
        idx = h_test_fold[qid] - 1
        # print >> l_test_out[idx], line
        for i in xrange(len(l_train_out)):
            if i == idx:
                print >> l_test_out[idx], line
                continue
            if with_dev:
                if i == ((idx + 1) % (len(l_train_out))):
                    print >> l_dev_out[i], line
                    continue
            print >> l_train_out[i], line

    for i in xrange(nb_folds):
        l_train_out[i].close()
        l_test_out[i].close()
    else_out.close()
    total_train_out.close()
    logging.info('all finished')

    return


if __name__ == '__main__':
    import sys
    if 5 > len(sys.argv):
        print 'I partition svm data in to k folds, sequentially'
        print '4+ para: input svm data + output dir + q st + q ed (include) + k (default 5) + with_dev (default 0)'
        sys.exit()
    q_st = int(sys.argv[3])
    q_ed = int(sys.argv[4])
    nb_folds = default_K
    if len(sys.argv) > 5:
        nb_folds = int(sys.argv[5])
    with_dev = False
    if len(sys.argv) > 6:
        with_dev = bool(int(sys.argv[6]))
    kfold_svm_data(sys.argv[1], q_st, q_ed, sys.argv[2], nb_folds, with_dev)








