import logging
import os
import subprocess
from os import path

from traitlets import Unicode, Tuple, Int, List
from traitlets.config import Configurable, Config

from knowledge4ir.model.hyper_para import HyperParameter
from knowledge4ir.utils import load_py_config, GDEVAL_PATH
from knowledge4ir.utils.model import fix_kfold_partition
from knowledge4ir.joint.model.attention_les import AttentionLes

h_model_name = {
    "att_les": AttentionLes
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

        self.model.train(train_x, train_y, self.l_hyper_para[0])
        logging.info('trained')
        self._dump_and_evaluate(test_x, out_dir, fold_k)
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

        self.model.train_with_dev(train_x, train_y, dev_x, dev_y, self.l_hyper_para)
        logging.info('trained')
        self._dump_and_evaluate(test_x, out_dir, fold_k)
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
        self.model.train(train_x, train_y, self.l_hyper_para[0])
        logging.info('trained')
        self._dump_and_evaluate(test_x, out_dir)
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
