"""
attention learning to rank model

The base class with input generators, and virtual methods
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    Float,
)
import json
import logging
import numpy as np
from knowledge4ir.model.base import pair_docno
from keras.callbacks import EarlyStopping
from knowledge4ir.utils import (
    group_scores_to_ranking,
    GDEVAL_PATH,
    dump_trec_ranking_with_score,
    seg_gdeval_out,
)
import subprocess


class AttLeToR(Configurable):
    train_in = Unicode(help="opt: trainning input").tag(config=True)
    dev_in = Unicode(help="opt: dev input").tag(config=True)
    test_in = Unicode(help="opt: testing input").tag(config=True)
    qrel = Unicode(help="qrel input").tag(config=True)

    out_dir = Unicode(help='output directory').tag(config=True)
    l2_w = Float(0, help='l2 regulalizer').tag(config=True)
    nb_q_t = Int(help='q term len').tag(config=True)
    nb_q_e = Int(help='q e len').tag(config=True)
    qt_rank_feature_dim = Int(help='q term ranking feature dim').tag(config=True)
    qe_rank_feature_dim = Int(help='q entity ranking feature dim').tag(config=True)
    qt_att_feature_dim = Int(help='q term attention feature dim').tag(config=True)
    qe_att_feature_dim = Int(help='q entity attention feature dim').tag(config=True)
    early_stop_patient = Int(10, help='early stopping patients').tag(config=True)

    qt_rank_name = Unicode('qt_rank')
    qt_att_name = Unicode('qt_att')
    qe_rank_name = Unicode('qe_rank')
    qe_att_name = Unicode('qe_att')
    aux_pre = Unicode('aux_')
    batch_size = Int(-1, help='batch size, if non-stochastic use -1').tag(config=True)

    nb_rank_layer = Int(1).tag(config=True)
    nb_att_layer = Int(1).tag(config=True)

    nb_epoch = Int(100, help='nb of epoch').tag(config=True)

    def __init__(self, **kwargs):
        super(AttLeToR, self).__init__(**kwargs)
        self.l_model_names = [self.qt_rank_name, self.qe_rank_name, self.qt_att_name, self.qe_att_name]
        self.l_input_dim = [(self.nb_q_t, self.qt_rank_feature_dim),
                            (self.nb_q_e, self.qe_rank_feature_dim),
                            (self.nb_q_t, self.qt_att_feature_dim),
                            (self.nb_q_e, self.qe_att_feature_dim),
                            ]
        self.ranking_model = None
        self.training_model = None

    def init_model(self):
        logging.info('initializing model')
        self.ranking_model, self.training_model = self._build_model()
        self.training_model.compile(
            optimizer='rmsprop',
            loss='hinge',
            # metric=['accuracy']
        )
        logging.info('ranking model summary')
        self.ranking_model.summary()
        logging.info('training model summary')
        self.training_model.summary()
        logging.info('model initialized')
        return

    def set_para(self, h_para):
        self.l2_w = h_para.get('l2_w', self.l2_w)
        logging.info('set para l2_w=[%f]', self.l2_w)
        return

    def train(self, train_lines=None, dev_lines=None):
        self.init_model()
        if train_lines is None:
            train_lines = open(self.train_in).read().splitlines()
        # if dev_lines is None:
        #     dev_lines = open(self.dev_in).read().splitlines()
        train_x, train_y = self.pairwise_construct(train_lines)
        logging.info('start training with [%d] pairs', train_y.shape[0])
        batch_size = self.batch_size
        if -1 == batch_size:
            batch_size = train_y.shape[0]
        if dev_lines:
            dev_x, dev_y = self.pairwise_construct(dev_lines)
            logging.info('with [%d] dev pairs', dev_y.shape[0])
            self.training_model.fit(
                train_x, train_y,
                batch_size=batch_size,
                nb_epoch=self.nb_epoch,
                validation_Data=(dev_x, dev_y),
                callbacks=[EarlyStopping(monitor='val_loss', patience=self.early_stop_patient)]
            )
        else:
            self.training_model.fit(
                train_x, train_y,
                batch_size=batch_size,
                nb_epoch=self.nb_epoch,
                callbacks=[EarlyStopping(monitor='loss', patience=self.early_stop_patient)]
            )

    def predict(self, test_lines=None):
        if not test_lines:
            test_lines = open(self.test_in).read().splitlines()
        assert self.training_model
        logging.info('start predicting')
        h_data, v_label = self.pointwise_construct(test_lines)
        l_qid, l_docno = self.get_qid_docno(test_lines)
        score = self.ranking_model.predict(h_data)
        l_score = score.reshape(-1).tolist()
        l_q_ranking = group_scores_to_ranking(l_qid, l_docno, l_score)
        logging.info('predicted')
        return l_q_ranking

    def evaluate(self, test_lines=None, out_pre=None):
        if not out_pre:
            out_pre = self.out_dir + '/run'
        l_q_ranking = self.predict(test_lines)
        dump_trec_ranking_with_score(l_q_ranking, out_pre + '.trec')
        eva_out = subprocess.check_output(['perl', GDEVAL_PATH, self.qrel, out_pre + '.trec'])
        print >> open(out_pre + '.eval', eva_out.strip())
        __, ndcg, err = seg_gdeval_out(eva_out, with_mean=True)
        logging.info('evaluated ndcg:%f, err:%f', ndcg, err)
        return ndcg

    def _build_model(self):
        """

        :return: ranker, pairwise_trainer
        """
        yield NotImplementedError

    def pointwise_construct(self, lines):
        l_qt_rank = []
        l_qe_rank = []
        l_qt_att = []
        l_qe_att = []
        l_y = []

        for line in lines:
            h = json.loads(line)
            l_feature_matrices = h['feature']
            y = h['rel']
            qt_rank_mtx, qe_rank_mtx, qt_att_mtx, qe_att_mtx = l_feature_matrices
            l_qt_rank.append(qt_rank_mtx)
            l_qe_rank.append(qe_rank_mtx)
            l_qt_att.append(qt_att_mtx)
            l_qe_att.append(qe_att_mtx)
            l_y.append(y)

            # if (batch_size != -1) & (len(l_y) >= batch_size):
            #     X = dict()
            #     X[self.qt_rank_name] = np.array(l_qt_rank)
            #     X[self.qe_rank_name] = np.array(l_qe_rank)
            #     X[self.qt_att_name] = np.array(l_qt_att)
            #     X[self.qe_att_name] = np.array(l_qe_att)
            #     Y = np.array(l_y)
            #     yield X, Y
            #     l_qt_rank, l_qe_rank, l_qt_att, l_qe_att = [], [], [], []
            #     l_y = []

        X = dict()
        X[self.qt_rank_name] = np.array(l_qt_rank)
        X[self.qe_rank_name] = np.array(l_qe_rank)
        X[self.qt_att_name] = np.array(l_qt_att)
        X[self.qe_att_name] = np.array(l_qe_att)
        Y = np.array(l_y)
        logging.info('[%d] pointwise data constructed finished', Y.shape[0])
        return X, Y

    @classmethod
    def get_qid_docno(cls, lines):
        l_qid = []
        l_docno = []
        for line in lines:
            h = json.loads(line)
            qid = h['q']
            docno = h['doc']
            l_qid.append(qid)
            l_docno.append(docno)
        return l_qid, l_docno

    def pairwise_construct(self, lines):
        """
        will read all data in memory first
        :param lines:
        :return:
        """

        point_X, point_Y = self.pointwise_construct(lines)

        l_qid, l_docno = self.get_qid_docno(lines)

        v_paired_label, l_paired_qid, l_docno_pair, l_pos_pair = pair_docno(point_Y, l_qid, l_docno)
        logging.info('total [%d] pairwise pair', len(l_paired_qid))
        i = 0

        l_left_pos = [item[0] for item in l_pos_pair]
        l_right_pos = [item[1] for item in l_pos_pair]

        qt_rank_mtx = point_X[self.qt_rank_name][l_left_pos, :]
        aux_qt_rank_mtx = point_X[self.qt_rank_name][l_right_pos, :]

        qe_rank_mtx = point_X[self.qe_rank_name][l_left_pos, :]
        aux_qe_rank_mtx = point_X[self.qe_rank_name][l_right_pos, :]

        qt_att_mtx = point_X[self.qt_att_name][l_left_pos, :]
        aux_qt_att_mtx = point_X[self.qt_att_name][l_right_pos, :]

        qe_att_mtx = point_X[self.qe_att_name][l_left_pos, :]
        aux_qe_att_mtx = point_X[self.qe_att_name][l_right_pos, :]

        X = dict()
        X[self.qt_rank_name] = qt_rank_mtx
        X[self.qe_rank_name] = qe_rank_mtx
        X[self.qt_att_name] = qt_att_mtx
        X[self.qe_att_name] = qe_att_mtx
        X[self.aux_pre + self.qt_rank_name] = aux_qt_rank_mtx
        X[self.aux_pre + self.qe_rank_name] = aux_qe_rank_mtx
        X[self.aux_pre + self.qt_att_name] = aux_qt_att_mtx
        X[self.aux_pre + self.qe_att_name] = aux_qe_att_mtx
        Y = v_paired_label

        return X, Y












