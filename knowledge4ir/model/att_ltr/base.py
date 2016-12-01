"""
attention learning to rank model

The base class with input generators, and virtual methods
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
    Int,
    Float,
)
import json
import logging
import numpy as np
from keras.layers import (
    Dense,
    Merge,
    Input,
    Activation,
    Flatten
)
from keras.models import (
    Model,
    Sequential,
)
from keras.callbacks import EarlyStopping
from knowledge4ir.model.base import pair_docno


class AttLeToR(Configurable):
    train_in = Unicode().tag(configure=True)
    dev_in = Unicode().tag(configure=True)
    test_in = Unicode().tag(configure=True)

    out_dir = Unicode(help='output directory').tag(configure=True)
    l2_w = Float(0, help='l2 regulalizer').tag(configure=True)
    nb_q_t = Int(help='q term len').tag(configure=True)
    nb_q_e = Int(help='q e len').tag(configure=True)
    qt_rank_feature_dim = Int(help='q term ranking feature dim').tag(configure=True)
    qe_rank_feature_dim = Int(help='q entity ranking feature dim').tag(configure=True)
    qt_att_feature_dim = Int(help='q term attention feature dim').tag(configure=True)
    qe_att_feature_dim = Int(help='q entity attention feature dim').tag(configure=True)

    qt_rank_name = Unicode('qt_rank')
    qt_att_name = Unicode('qt_att')
    qe_rank_name = Unicode('qe_rank')
    qe_att_name = Unicode('qe_att')
    aux_pre = Unicode('aux_')

    nb_rank_layer = Int(1).tag(configure=True)
    nb_att_layer = Int(1).tag(configure=True)

    nb_epoch = Int(100, help='nb of epoch').tag(configure=True)

    def pointwise_read(self, in_name):
        l_qt_rank = []
        l_qe_rank = []
        l_qt_att = []
        l_qe_att = []
        l_y = []

        for line in open(in_name):
            h = json.loads(line)
            qid = h['q']
            docno = h['docno']
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
        logging.info('pointwise finished')
        return X, Y

    def pairwise_read(self, in_name):
        """
        will read all data in memory first
        :param in_name:
        :return:
        """

        point_X, point_Y = self.pointwise_read(in_name, -1)

        l_qid, l_docno = [], []
        for line in open(in_name):
            h = json.loads(line)
            qid = h['q']
            docno = h['docno']
            l_qid.append(qid)
            l_docno.append(docno)

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












