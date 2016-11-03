"""
run cross validation using rank lib
do:
    for each fold:
        train -> re-rank test -> merge to a total trec ranking data for the kfold data
    for total:
        train on the total train, and re-rank rest data, -> merge to trec format
        if the total_train.txt and else.txt data exists
the folder dir must follow letor 3.0 convention
ranklib is given as a Jar
ranking model is given as a number
will dump ranklib's logs in to output folder
the intermediate data and final data is also in the output folder

only use class in order the easy configuration
"""

import json
import logging
import os
import subprocess

from traitlets import (
    Int,
    Unicode,
    List,
    Float,
    Bool,
)
from traitlets.config import Configurable

from knowledge4ir.utils import ROOT_PATH
from knowledge4ir.utils import (
    dump_trec_out_from_ranking_score,
)
from knowledge4ir.utils.evaluation import GDEVAL_PATH



class RanklibRunner(Configurable):
    ranklib = Unicode(ROOT_PATH + '/knowledge4ir/letor/RankLib.jar',
                      help='the location of ranklib jar'
                      ).tag(config=True)
    nb_fold = Int(5, help='fold number k').tag(config=True)
    fold_dir = Unicode(help='input fold dir').tag(config=True)
    out_dir = Unicode(help='output dir').tag(config=True)
    model_id = Unicode('4', help='model id as defined in ranklib, -1==ranksvm, -2==hybrid').tag(config=True)
    qrel = Unicode(help='qrel path').tag(config=True)
    ranksvm = Unicode(ROOT_PATH + '/knowledge4ir/letor/rank_svm',
                      help='the location of ranksvm bin file'
                      ).tag(config=True)
    ranksvm_c = Float(0.1, help='C of ranksvm').tag(config=True)
    l_ranksvm_c = List(Float, default_value=[0.00001, 0.0001, 0.001, 0.01, 0.03, 0.05, 0.07, 0.1],
                       help='the ranksvm c range for cross validate',
                       ).tag(config=True)
    with_dev = Bool(False, help='tune parameter with development, only support rank svm now'
                    ).tag(config=True)

    def __init__(self, **kwargs):
        super(RanklibRunner, self).__init__(**kwargs)
        self.l_ranklib_cmb = ['java', '-jar', self.ranklib, '-ranker', self.model_id,
                              '-metric2t', 'map',
                              '-metric2T', 'NDCG@20'
                              ]
        self.l_ranksvm_learn = [self.ranksvm + '_learn']
        self.l_ranksvm_pre = [self.ranksvm + '_classify']
        self.l_fold_dir = []
        self.l_out_fold_dir = []
        self.train_name = 'train.txt'
        self.test_name = 'test.txt'
        self.dev_name = 'dev.txt'
        self.predict_name = 'predict'
        self.rank_name = 'trec'
        self.eval_name = 'eval'
        self.log_name = 'log'
        if self.with_dev:
            assert self.model_id == '-1'

    def cross_validation(self, fold_dir=None, out_dir=None):
        """
        run cross_validation
        :return:
        """
        if fold_dir:
            self.fold_dir = fold_dir
        if out_dir:
            self.out_dir = out_dir
        self._form_fold_dir()
        if self.with_dev:
            self._cross_validation_with_dev()
        else:
            self._cross_validation_without_dev()
        return

    def _cross_validation_without_dev(self):
        for indir, subdir in zip(self.l_fold_dir, self.l_out_fold_dir):
            train_in = os.path.join(indir, self.train_name)
            test_in = os.path.join(indir, self.test_name)
            score_out = os.path.join(subdir, self.predict_name)
            log_out = os.path.join(subdir, self.log_name)
            self._train_test(train_in, test_in, score_out, log_out)
        self._merge_evaluate_trec_rank()
        logging.info('cross validation finished')
        return

    def _cross_validation_with_dev(self):
        for indir, subdir in zip(self.l_fold_dir, self.l_out_fold_dir):
            train_in = os.path.join(indir, self.train_name)
            test_in = os.path.join(indir, self.test_name)
            dev_in = os.path.join(indir, self.dev_name)
            score_out = os.path.join(subdir, self.predict_name)
            log_out = os.path.join(subdir, self.log_name)
            self._train_dev_test(train_in, dev_in, test_in, score_out, log_out)
        self._merge_evaluate_trec_rank()
        logging.info('cross validation with development finished')
        return

    def total_train_test(self):
        self._form_fold_dir()
        train_in = os.path.join(self.fold_dir, 'total_train.txt')
        test_in = os.path.join(self.fold_dir, 'else.txt')
        score_out = os.path.join(self.out_dir, 'else.' + self.predict_name)
        log_out = os.path.join(self.out_dir, 'else.' + self.log_name)
        self._train_test(train_in,
                         test_in,
                         score_out,
                         log_out)
        trec_out = os.path.join(self.out_dir, 'else.' + self.rank_name)
        self._form_trec_rank(test_in, score_out, trec_out)
        return

    def _train_test(self, train_in, test_in, score_out, log_out):
        if self.model_id == '-1':
            return self._train_test_ranksvm(train_in, test_in, score_out, log_out)
        else:
            return self._train_test_ranklib(train_in, test_in, score_out, log_out)

    def _train_test_ranklib(self, train_in, test_in, score_out, log_out):
        l_train_cmd = list(self.l_ranklib_cmb)
        l_train_cmd.extend(['-train',
                            train_in,
                            '-test',
                            test_in,
                            '-save',
                            score_out + '.model'
                            ])
        logging.info('running %s', json.dumps(l_train_cmd))
        out_str = subprocess.check_output(l_train_cmd)
        print >> open(log_out, 'w'), out_str
        logging.info('training finished with [%s]', out_str.split('\n')[-3:])

        l_rank_cmd = list(self.l_ranklib_cmb)
        l_rank_cmd.extend(['-load',
                           score_out + '.model',
                           '-rank',
                           test_in,
                           '-score',
                           score_out
                           ])
        out_str = subprocess.check_output(l_rank_cmd)
        logging.info('reranking finished with [%s]', out_str.split('\n')[-3:])

        return

    def _train_test_ranksvm(self, train_in, test_in, score_out, log_out):
        l_train_cmd = list(self.l_ranksvm_learn)
        l_train_cmd.extend(['-c', '%f' % self.ranksvm_c, train_in,
                            score_out + '.model'
                            ])
        logging.info('running %s', json.dumps(l_train_cmd))
        out_str = subprocess.check_output(l_train_cmd)
        print >> open(log_out, 'w'), out_str
        logging.info('training finished with [%s]', out_str.split('\n')[-3:])

        l_rank_cmd = list(self.l_ranksvm_pre)
        l_rank_cmd.extend([test_in,
                           score_out + '.model',
                           score_out
                           ])
        out_str = subprocess.check_output(l_rank_cmd)
        logging.info('reranking finished with [%s]', out_str.split('\n')[-3:])
        return

    def _train_dev_ranksvm(self, train_in, dev_in, svm_c):
        self.ranksvm_c = svm_c
        dev_out = dev_in + '_pre'
        self._train_test_ranksvm(train_in, dev_in, dev_out, dev_out + '_log')
        self._form_trec_rank(dev_in, dev_out, dev_out + '.trec')
        eva_str = subprocess.check_output(['perl',
                                           GDEVAL_PATH,
                                           self.qrel, dev_out + '.trec'])
        print >> open(dev_out + '.eval', 'w'), eva_str
        ndcg = self._seg_mean_ndcg(eva_str)
        logging.info('dev [%s] c [%f] got ndcg [%f]', dev_in, svm_c, ndcg)
        return ndcg

    def _combine_train_test_ranksvm(self, train_in, dev_in, test_in, svm_c, score_out, log_out):
        total_train_in = train_in + '.plus_dev'
        out = open(total_train_in, 'w')
        lines = open(train_in).read().splitlines() + open(dev_in).read().splitlines()
        lines.sort(key=lambda item: int(item.split()[1].replace('qid:', '')))
        print >> out, '\n'.join(lines)
        out.close()
        self.ranksvm_c = svm_c
        return self._train_test_ranksvm(total_train_in, test_in, score_out, log_out)

    def _train_dev_test(self, train_in, dev_in, test_in, score_out, log_out):
        """
        pick the best c using dev
        and then train test
        :param train_in:
        :param dev_in:
        :param test_in:
        :param score_out:
        :param log_out:
        :return:
        """
        best_c = None
        best_ndcg = None
        logging.info('start train [%s] dev test', train_in)
        for svm_c in self.l_ranksvm_c:
            this_ndcg = self._train_dev_ranksvm(train_in, dev_in, svm_c)
            if best_c is None:
                logging.info('start with [%f-%f]', svm_c, this_ndcg)
                best_c = svm_c
                best_ndcg = this_ndcg
                continue
            if this_ndcg > best_ndcg:
                logging.info('improved to [%f-%f]', svm_c, this_ndcg)
                best_c = svm_c
                best_ndcg = this_ndcg
        logging.info('best dev c: [%f-%f]', best_c, best_ndcg)
        self._combine_train_test_ranksvm(train_in, dev_in, test_in, best_c, score_out, log_out)
        logging.info('train [%s] dev test finished', train_in)
        return


    @classmethod
    def _seg_mean_ndcg(cls, eva_str):
        line = eva_str.splitlines()[-1]
        ndcg, err = line.split(',')[-2:]
        ndcg = float(ndcg)
        return ndcg

    @classmethod
    def _form_trec_rank(cls, test_in, pre_out, trec_out):
        """
        form trec format output ranking
        :param test_in:
        :param pre_out:
        :param trec_out:
        :return:
        """
        l_lines = open(test_in).read().splitlines()

        l_qid = [line.split()[1].replace('qid:', "") for line in l_lines]
        l_docno = [line.split('#')[-1].strip() for line in l_lines]
        l_score = [float(line.strip().split()[-1]) for line in open(pre_out)]

        dump_trec_out_from_ranking_score(l_qid, l_docno, l_score, trec_out, 'ranklib')
        return

    def _merge_evaluate_trec_rank(self):
        """
        merge to a final trec rank
        and evaluate using gdeval
        :return:
        """
        total_trec_out_name = os.path.join(self.out_dir, self.rank_name)
        total_trec_out = open(total_trec_out_name, 'w')
        for indir, subdir in zip(self.l_fold_dir, self.l_out_fold_dir):
            pre_out = os.path.join(subdir, self.predict_name)
            test_in = os.path.join(indir, self.test_name)
            trec_out = os.path.join(subdir, self.rank_name)
            self._form_trec_rank(test_in, pre_out, trec_out)
            lines = open(trec_out).read()
            print >> total_trec_out, lines.strip()
        total_trec_out.close()

        for d in [1, 3, 5, 10, 20]:
            eva_str = subprocess.check_output(['perl',  GDEVAL_PATH, '-k', '%d' % d, self.qrel, total_trec_out_name])
            eva_out = os.path.join(self.out_dir, self.eval_name + '.d%02d' % d)
            print >> open(eva_out, 'w'), eva_str.strip()
        eva_str = subprocess.check_output(['perl', GDEVAL_PATH, self.qrel, total_trec_out_name])
        eva_out = os.path.join(self.out_dir, self.eval_name)
        print >> open(eva_out, 'w'), eva_str.strip()
        ndcg, err = eva_str.strip().splitlines()[-1].split(',')[-2:]
        ndcg = float(ndcg)
        err = float(err)
        logging.info('cv evaluation: %s', eva_str.strip().splitlines()[-1])

        return ndcg

    def _form_fold_dir(self):
        self.l_fold_dir = []
        self.l_out_fold_dir = []
        for k in xrange(self.nb_fold):
            self.l_fold_dir.append(os.path.join(self.fold_dir, 'Fold%d' % (k + 1)))
            self.l_out_fold_dir.append(os.path.join(self.out_dir, 'Fold%d' % (k + 1)))

        for dirname in self.l_fold_dir + self.l_out_fold_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import load_py_config, set_basic_log

    set_basic_log()
    if 2 > len(sys.argv):
        print "I run ranklib for cv"
        print '1+ para config + (opt)cv|rerank'
        RanklibRunner.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])
    runner = RanklibRunner(config=conf)
    method = 'cv'
    if len(sys.argv) >= 3:
        method = sys.argv[2]
    if method == 'rerank':
        runner.total_train_test()
    else:
        runner.cross_validation()
