"""
cross validate hybrid model
"""

from scholarranking.letor.hybrid_model.model import HybridLeToR
from scholarranking.letor import fix_kfold_partition
from scholarranking.letor import filter_svm_data
from scholarranking.utils import (
    load_py_config,
    load_svm_feature,
    dump_trec_ranking_with_score,
    GDEVAL_PATH,
    QREL_PATH,
    seg_gdeval_out,
    set_basic_log,
)
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode,
    List,
    Bool,
    Dict,
)
import logging
import json
import os
import subprocess


class CrossValidator(Configurable):
    svm_data_in = Unicode(help="total data in").tag(config=True)
    with_dev = Bool(True, help='with development').tag(config=True)
    h_dev_para = Dict(default_value={'l2_w': [0, 0.01, 0.1, 1]},
                      help="to explore parameters").tag(config=True)
    out_dir = Unicode(help="out dir").tag(config=True)

    nb_folds = Int(10, help="k").tag(config=True)
    q_st = Int(1)
    q_ed = Int(100)

    def __init__(self, **kwargs):
        super(CrossValidator, self).__init__(**kwargs)
        self.model = HybridLeToR(**kwargs)
        self.l_train_folds, self.l_test_folds, self.l_dev_folds = fix_kfold_partition(
                self.with_dev, self.nb_folds, self.q_st, self.q_ed
            )
        self.l_svm_data = load_svm_feature(self.svm_data_in)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    @classmethod
    def class_print_help(cls, inst=None):
        super(CrossValidator, cls).class_print_help(inst)
        HybridLeToR.class_print_help(inst)

    def train_test_fold(self, k):
        out_dir = os.path.join(self.out_dir, 'Fold%d' % k)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        l_train_svm = filter_svm_data(self.l_svm_data, self.l_train_folds[k])
        l_test_svm = filter_svm_data(self.l_svm_data, self.l_test_folds[k])
        self.model.train(l_train_svm)
        l_q_ranking = self.model.predict(l_test_svm)
        rank_out_name = out_dir + '/trec'
        eva_out_name = out_dir + '/eval'
        dump_trec_ranking_with_score(l_q_ranking, rank_out_name)
        eva_str = subprocess.check_output(
            ['perl', GDEVAL_PATH, QREL_PATH, rank_out_name]).strip()
        print >> open(eva_out_name, 'w'), eva_str.strip()
        logging.info("training testing fold %d done with %s",
                     k, eva_str.splitlines()[-1])
        return

    def train_dev_test_fold(self, k):
        out_dir = os.path.join(self.out_dir, 'Fold%d' % k)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        l_train_svm = filter_svm_data(self.l_svm_data, self.l_train_folds[k])
        l_test_svm = filter_svm_data(self.l_svm_data, self.l_test_folds[k])
        l_dev_svm = filter_svm_data(self.l_svm_data, self.l_dev_folds[k])
        best_ndcg = 0
        best_para = None
        dev_eva_out = open(out_dir + '/dev_para.eval', 'w')
        logging.info('start developing parameters')
        for h_para in self._dev_para_generator():
            logging.info('evaluating para %s', json.dumps(h_para))
            self.model.set_para(h_para)
            self.model.train(l_train_svm)
            l_q_ranking = self.model.predict(l_dev_svm)
            rank_out_name = out_dir + '/dev.trec'
            dump_trec_ranking_with_score(l_q_ranking, rank_out_name)
            eva_str = subprocess.check_output(
                ['perl', GDEVAL_PATH, QREL_PATH, rank_out_name]).strip()
            __, ndcg, err = seg_gdeval_out(eva_str)
            logging.info('para %s get ndcg %f', json.dumps(h_para), ndcg)
            print >> dev_eva_out, '%s\t%f,%f' % (json.dumps(h_para), ndcg, err)
            if ndcg > best_ndcg:
                logging.info('get better ndcg %f with %s', ndcg, json.dumps(h_para))
                best_ndcg = ndcg
                best_para = h_para
        dev_eva_out.close()
        logging.info('best ndcg %f with %s', best_ndcg, json.dumps(best_para))
        logging.info('start training total')
        self.model.set_para(best_para)
        self.model.train(l_train_svm + l_dev_svm)
        l_q_ranking = self.model.predict(l_test_svm)
        rank_out_name = out_dir + '/trec'
        eva_out_name = out_dir + '/eval'
        dump_trec_ranking_with_score(l_q_ranking, rank_out_name)
        eva_str = subprocess.check_output(
            ['perl', GDEVAL_PATH, QREL_PATH, rank_out_name]).strip()
        print >> open(eva_out_name, 'w'), eva_str.strip()
        __, ndcg, err = seg_gdeval_out(eva_str)
        logging.info('training testing fold %d done with ndcg %f', k, ndcg)
        return

    def run_one_fold(self, k):
        if self.with_dev:
            self.train_dev_test_fold(k)
        else:
            self.train_test_fold(k)

    def _dev_para_generator(self):
        for l2_w in self.h_dev_para['l2_w']:
            h_para = {'l2_w': l2_w}
            yield h_para

if __name__ == '__main__':
    import sys
    set_basic_log()
    if 3 != len(sys.argv):
        print "cross validate one fold"
        print '2 para: config + fold k'
        CrossValidator.class_print_help()
        sys.exit()

    k = int(sys.argv[2])
    conf = load_py_config(sys.argv[1])
    runner = CrossValidator(config=conf)
    runner.run_one_fold(k)











