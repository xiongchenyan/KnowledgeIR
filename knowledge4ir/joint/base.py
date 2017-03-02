"""
model's hyper parameters
base functions
"""

from traitlets.config import Configurable
from traitlets import (
    Float,
    Int,
    Tuple,
    Unicode,
    List,
)
from knowledge4ir.joint.resource import JointSemanticResource
import logging
import json
from keras.callbacks import EarlyStopping
import math


class HyperParameter(Configurable):
    # model parameters
    l2_w = Float(0.01).tag(config=True)
    dropout_rate = Float(0).tag(config=True)
    q_shape = Tuple(Int, default_value=(5, 1)).tag(config=True)
    title_shape = Tuple(Int, default_value=(10, 1)).tag(config=True)
    body_shape = Tuple(Int, default_value=(300, 1)).tag(config=True)
    embedding_dim = Int(300).tag(config=True)
    l_kernel_pool_mean =List(Float, default_value=[],
                             help='will always add the exact kernel'
                             ).tag(config=True)
    kernel_pool_lambda = Float(0.1).tag(config=True)

    # training parameters
    loss = Unicode('hinge').tag(config=True)
    opt = Unicode('nadam').tag(config=True)
    batch_size = Int(-1).tag(config=True)
    nb_epoch = Int(10).tag(config=True)
    early_stopping_patient = Int(10).tag(config=True)


    def pretty_print(self):
        #TODO
        return


class JointSemanticModel(Configurable):
    """
    the base class for all models
    """
    aux_pre = Unicode('aux_')
    q_name = Unicode('q')
    q_att = Unicode('q_att')
    title_name = Unicode('title')
    body_name = Unicode('body')
    title_att = Unicode('title_att')
    body_att = Unicode('body_att')

    def __init__(self, **kwargs):
        super(JointSemanticModel, self).__init__(**kwargs)
        self.hyper_para = HyperParameter(**kwargs)
        self.external_resource = JointSemanticResource(**kwargs)

        self.ranking_model = None
        self.training_model = None

    def set_external_resource(self, resource):
        self.external_resource = resource

    def _build_model(self):
        h_para_layers = self._build_para_layers()
        self.ranking_model, self.training_model = self._form_model_from_layers(h_para_layers)
        return

    def _build_para_layers(self):
        raise NotImplementedError

    def _form_model_from_layers(self, h_para_layers):
        raise NotImplementedError

    def pairwise_train(self, l_paired_data, Y, hyper_para=None):
        """
        pairwise training
        :param l_paired_data: each element is a pair of doc's, Y is their order (1 or -1, for the order)
        :param Y: label
        :param hyper_para: if set, then use this one
        :return: trained model
        """
        if not hyper_para:
            hyper_para = self.hyper_para
        logging.info('training with para: %s', hyper_para.pretty_print())
        batch_size = hyper_para.batch_size
        if -1 == batch_size:
            batch_size = len(l_paired_data)
        self._build_model()
        self.training_model.compile(
            hyper_para.opt,
            hyper_para.loss,
        )

        logging.info('start training with [%d] data with full batch', len(l_paired_data))

        self.training_model.fit(
            l_paired_data,
            Y,
            batch_size=batch_size,
            nb_epoch=hyper_para.nb_epoch,
            callbacks = EarlyStopping(monitor='loss',
                                      patience=self.hyper_para.early_stopping_patient
                                      ),
        )
        logging.info('model training finished')
        return

    def predict(self, l_data):
        """
        assume the model is trained
        :param l_data:
        :return:
        """
        Y = self.ranking_model.predict(l_data, batch_size=len(l_data))
        return Y

    def hyper_para_dev(self, l_paired_train, Y, l_paired_dev, Y_dev, l_hyper_para):
        """
        return the best hyper_pra in l_hyper_para, based on performance in dev data
        :param l_paired_train:
        :param Y:
        :param l_paired_dev:
        :param Y_dev:
        :param l_hyper_para:
        :return:
        """
        best_loss = None
        best_para = None
        for hyper_para in l_hyper_para:
            self.pairwise_train(l_paired_train, Y, hyper_para)
            this_loss = self.ranking_model.evaluate(l_paired_dev, Y_dev)
            if not best_loss:
                best_para = hyper_para
                best_loss = this_loss
            if best_loss > this_loss:
                best_loss = this_loss
                best_para = hyper_para
            logging.info('dev fold this loss [%f] best one [%f]', this_loss, best_loss)

        return best_para

    def pairwise_train_with_dev(self, l_paired_data, Y, l_hyper_para, dev_frac=0.1):
        ed = math.ceil(len(l_paired_data) * (1 - dev_frac))
        l_paired_train = l_paired_data[:ed]
        l_paired_dev = l_paired_data[ed:]
        Y_train = Y[:ed]
        Y_dev = Y[ed:]

        best_para = self.hyper_para_dev(l_paired_train, Y_train,
                                        l_paired_dev, Y_dev,
                                        l_hyper_para)
        logging.info('best para in dev: %s', best_para.pretty_print())

        self.pairwise_train(l_paired_data, Y, best_para)
        return




