import logging
import os
import subprocess
from os import path

from traitlets import Unicode, Tuple, Int, List
from traitlets.config import Configurable, Config

from knowledge4ir.model.hyper_para import HyperParameter
from knowledge4ir.utils import load_py_config, GDEVAL_PATH
from knowledge4ir.utils.model import fix_kfold_partition
from knowledge4ir.joint.model.attention_les import (
    AttentionLes,
    Les,
    DisAmbiAttLes,
    SfAttLes,
)
import json

h_model_name = {
    "att_les": AttentionLes,
    "les": Les,
    "att_e_les": DisAmbiAttLes,
    "att_sf_les": SfAttLes,
}


class CrossValidator(Configurable):
    nb_folds = Int(10).tag(config=True)
    model_name = Unicode('att_les', help="the model name").tag(config=True)
    model_conf = Unicode(help='the model config file').tag(config=True)
    q_range = List(Int, default_value=[1, 200]).tag(config=True)
    l_hyper_para_in = List(Unicode, help="the file names of hyper paras, if dev, then explore the list,"
                                         "if no dev, then the first one is the default "
                           ).tag(config=True)
    qrel_in = Unicode(help='qrel in').tag(config=True)
    nb_repeat = Int(
        1,
        help='number of repeat training time, will pick the one with best training loss to apply'
    ).tag(config=True)

    def __init__(self, **kwargs):
        super(CrossValidator, self).__init__(**kwargs)

        assert self.model_name in h_model_name
        logging.info('using model [%s] with conf in [%s]', self.model_name, self.model_conf)
        conf = Config()
        if self.model_conf:
            conf = load_py_config(self.model_conf)
        self.model = h_model_name[self.model_name](config=conf)
        logging.info('ranking model initialized')
        self.l_hyper_para = []
        self._load_hyper_para()

    def _load_hyper_para(self):
        self.l_hyper_para = [HyperParameter(config=load_py_config(para_in))
                             for para_in in self.l_hyper_para_in]
        logging.info('[%d] hyper parameters loaded', len(self.l_hyper_para))

    def train_test_one_fold(self, in_name, out_dir, fold_k):
        """
        output will be in out_dir/Foldk
            trec
            eval
        will use the first para in l_hyper_para_in
        :param in_name:
        :param out_dir:
        :param fold_k:
        :return:
        """
        logging.info('training and testing one fold for [%s][%d]', in_name, fold_k)
        l_train, l_test, __ = fix_kfold_partition(False, k=10,
                                                  st=self.q_range[0],
                                                  ed=self.q_range[1]
                                                  )

        s_train_qid = set(l_train[fold_k])
        s_test_qid = set(l_test[fold_k])
        train_x, train_y = self.model.train_data_reader(in_name, s_train_qid)
        test_x, _ = self.model.test_data_reader(in_name, s_test_qid)

        best_train_loss = None
        best_ndcg = None
        for p in xrange(self.nb_repeat):
            logging.info('repeating training [%d]', p)
            loss = self.model.train(train_x, train_y, self.l_hyper_para[0])
            logging.info('trained [%d] with loss [%f]', p, loss)
            if best_train_loss is not None:
                if best_train_loss < loss:
                    logging.info('no improvement in training loss [%f]>[%f],skip this training',
                                 loss, best_train_loss)
                    continue
            logging.info('get new best training loss %f vs %s, use it on testing data',
                         loss, json.dumps(best_train_loss))
            ndcg = self._dump_and_evaluate(test_x, out_dir, fold_k)
            logging.info('[%d] try, loss %s -> %f, test ndcg %s -> %f',
                         p, json.dumps(best_train_loss), loss,
                         json.dumps(best_ndcg), ndcg)
            best_train_loss, best_ndcg = loss, ndcg
        logging.info('[%s][%d] finished, loss [%f], ndcg [%f]', out_dir, fold_k,
                     best_train_loss, best_ndcg)
        return

    def train_dev_test_one_fold(self, in_name, out_dir, fold_k):
        """
        train-dev-test for one fold
        output will be in out_dir/Foldk
            trec
            eval
        :param in_name:
        :param out_dir:
        :param fold_k:
        :param l_para:
        :return:
        """
        logging.info('training, dev, and testing one fold for [%s][%d]', in_name, fold_k)
        l_train, l_test, l_dev = fix_kfold_partition(False, k=10,
                                                     st=self.q_range[0],
                                                     ed=self.q_range[1]
                                                     )

        s_train_qid = set(l_train[fold_k])
        s_test_qid = set(l_test[fold_k])
        s_dev_qid = set(l_dev[fold_k])
        train_x, train_y = self.model.train_data_reader(in_name, s_train_qid)
        dev_x, dev_y = self.model.train_data_reader(in_name, s_dev_qid)
        test_x = self.model.test_data_reader(in_name, s_test_qid)

        best_train_loss = None
        best_ndcg = None
        for p in xrange(self.nb_repeat):
            logging.info('repeating training [%d]', p)
            loss = self.model.train_with_dev(train_x, train_y, dev_x, dev_y, self.l_hyper_para)
            logging.info('trained p with loss [%f]', loss)
            if best_train_loss is not None:
                if best_train_loss < loss:
                    logging.info('no improvement in training loss [%f]>[%f],skip this training',
                                 loss, best_train_loss)
                    continue
            logging.info('get new best training loss %f vs %s, use it on testing data',
                         loss, json.dumps(best_train_loss))
            ndcg = self._dump_and_evaluate(test_x, out_dir, fold_k)
            logging.info('[%d] try, loss %s -> %f, test ndcg %s -> %f',
                         p, json.dumps(best_train_loss), loss,
                         json.dumps(best_ndcg), ndcg)
            best_train_loss, best_ndcg = loss, ndcg
        logging.info('[%s][%d] finished, loss [%f], ndcg [%f]', out_dir, fold_k,
                     best_train_loss, best_ndcg)
        return

    def train_test_files(self, train_in, test_in, out_dir):
        """

        :param train_in: file to train on
        :param test_in: file to test on
        :param out_dir: out directory
        :return:
        """
        logging.info('train and test with [%s] -> [%s]',
                     train_in, test_in)
        s_qid = ["%d" % i for i in range(self.q_range[0], self.q_range[1] + 1)]
        train_x, train_y = self.model.train_data_reader(train_in, s_qid)
        test_x, _ = self.model.test_data_reader(test_in, s_qid)

        best_train_loss = None
        best_ndcg = None
        for p in xrange(self.nb_repeat):
            logging.info('repeating training [%d]', p)
            loss = self.model.train(train_x, train_y, self.l_hyper_para[0])
            logging.info('trained p with loss [%f]', loss)
            if best_train_loss is not None:
                if best_train_loss < loss:
                    logging.info('no improvement in training loss [%f]>[%f],skip this training',
                                 loss, best_train_loss)
                    continue
            logging.info('get new best training loss %f vs %s, use it on testing data',
                         loss, json.dumps(best_train_loss))
            ndcg = self._dump_and_evaluate(test_x, out_dir)
            logging.info('[%d] try, loss %s -> %f, test ndcg %s -> %f',
                         p, json.dumps(best_train_loss), loss,
                         json.dumps(best_ndcg), ndcg)
            best_train_loss, best_ndcg = loss, ndcg
        logging.info('[%s] finished, loss [%f], ndcg [%f]', test_in,
                     best_train_loss, best_ndcg)
        return

    def train_test_generator(self, train_in, test_in, out_dir, s_train_qid, s_test_qid):
        """
        train and test with generator
        """
        logging.info('train and test with [%s] -> [%s]',
                     train_in, test_in)
        self.model.train_generator(train_in, self.l_hyper_para[0], s_train_qid)
        logging.info('trained')
        self._dump_and_evaluate_generator(test_in, out_dir, s_test_qid)
        return

    def _dump_and_evaluate(self, test_x, out_dir, fold_k=None):
        """
        self.model is trained
        :param test_x: test X
        :param out_dir: cv out dir
        :param fold_k: the current fold
        :return:
        """
        rank_out = self._form_rank_out_name(out_dir, fold_k)
        self.model.generate_ranking(test_x, rank_out)
        logging.info('ranking results to [%s]', rank_out)
        eva_res = subprocess.check_output(['perl', GDEVAL_PATH, self.qrel_in, rank_out])
        eva_out = self._form_eval_out_name(out_dir, fold_k)
        print >> open(eva_out, 'w'), eva_res.strip()
        if fold_k is not None:
            logging.info('fold [%d] finished to [%s], result [%s]',
                         fold_k, eva_out, eva_res.splitlines()[-1]
                         )
        else:
            logging.info('finished to [%s], result [%s]',
                         eva_out, eva_res.splitlines()[-1]
                         )
        ndcg, err = eva_res.splitlines()[-1].split(',')[-2:]
        ndcg = float(ndcg)
        return ndcg

    def _dump_and_evaluate_generator(self, test_in, out_dir, s_test_qid):
        """
        self.model is trained
        :param test_in: test input
        :param out_dir: cv out dir
        :param s_test_qid: the test qid set
        :return:
        """
        rank_out = self._form_rank_out_name(out_dir, None)
        self.model.generate_ranking_generator(test_in, rank_out, s_test_qid)
        logging.info('ranking results to [%s]', rank_out)
        eva_res = subprocess.check_output(['perl', GDEVAL_PATH, self.qrel_in, rank_out])
        eva_out = self._form_eval_out_name(out_dir, None)
        print >> open(eva_out, 'w'), eva_res.strip()
        logging.info('evaluation result dumped to [%s], result [%s]', eva_out, eva_res.splitlines()[-1])
        return

    @classmethod
    def _form_fold_dir(cls, out_dir, fold_k=None):
        if fold_k is not None:
            fold_dir = path.join(out_dir, 'Fold%d' % fold_k)
        else:
            fold_dir = path.join(out_dir, 'uni')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        return fold_dir

    @classmethod
    def _form_dev_out_name(cls, out_dir, fold_k=None):
        fold_dir = cls._form_fold_dir(out_dir, fold_k)
        out_name = path.join(fold_dir, 'dev')
        return out_name

    @classmethod
    def _form_rank_out_name(cls, out_dir, fold_k=None):
        fold_dir = cls._form_fold_dir(out_dir, fold_k)
        out_name = path.join(fold_dir, 'trec')
        return out_name

    @classmethod
    def _form_eval_out_name(cls, out_dir, fold_k=None):
        fold_dir = cls._form_fold_dir(out_dir, fold_k)
        out_name = path.join(fold_dir, 'eval')
        return out_name
