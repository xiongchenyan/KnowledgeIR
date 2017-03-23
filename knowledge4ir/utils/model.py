def fix_kfold_partition(with_dev=False, k=10, st=1, ed=200):
    l_train_folds = []
    l_dev_folds = []
    l_test_folds = []
    for fold in xrange(k):
        test = []
        train = []
        dev = []
        for qid in xrange(st, ed + 1):
            if (qid % k) == fold:
                test.append("%d" % qid)
                continue
            if with_dev:
                if ((qid + 1) % k) == fold:
                    dev.append("%d" % qid)
                    continue
            train.append("%d" % qid)
        l_train_folds.append(train)
        l_test_folds.append(test)
        l_dev_folds.append(dev)
    return l_train_folds, l_test_folds, l_dev_folds
