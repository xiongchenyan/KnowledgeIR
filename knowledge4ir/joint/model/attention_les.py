"""
les with attention (only on query entities)


model input:
    defined in model.base

a subclass of JointSemanticModel
only need to define the neural network model

"""

from keras.models import (
    Model,
    Sequential
)
from keras.layers import (
    Flatten,
    Merge,
    merge,
    Dense,
    Conv1D,
    Conv2D,
    Input,
    RepeatVector,
    Lambda,
    Reshape,
    Activation,
    Permute,
    AveragePooling1D,
)
from keras.regularizers import (
    l2
)

import keras.backend as K
import logging
import json
from knowledge4ir.joint.model import (
    l_input_name,
    sf_ground_name,
    e_match_name,
    e_ground_name,
    ltr_feature_name,
    JointSemanticModel
)
from traitlets import (
    Unicode,
    Int,
    List,
)
import numpy as np


class AttentionLes(JointSemanticModel):
    model_name = Unicode('att_les')
    max_spot_per_q = Int(3, help='max spot allowed per q').tag(config=True)
    max_e_per_spot = Int(3, help='top e allowed per q').tag(config=True)
    sf_ground_f_dim = Int(6, help='sf ground feature dimension').tag(config=True)
    e_ground_f_dim = Int(5, help='e ground feature dimension').tag(config=True)
    e_match_f_dim = Int(16, help='e match feature dimension').tag(config=True)
    ltr_f_dim = Int(1, help='ltr feature dimension').tag(config=True)
    l_x_name = List(Unicode, default_value=l_input_name).tag(config=True)
    e_att_activation = Unicode('linear', help='the activation on e grounding').tag(config=True)

    def __init__(self, **kwargs):
        super(AttentionLes, self).__init__(**kwargs)
        self.sf_ground_shape = (self.max_spot_per_q, self.sf_ground_f_dim)
        self.e_ground_shape = (self.max_spot_per_q, self.max_e_per_spot, self.e_ground_f_dim)
        self.e_match_shape = (self.max_spot_per_q, self.max_e_per_spot, self.e_match_f_dim)
        self.ltr_shape = (self.ltr_f_dim,)

    def pairwise_data_reader(self, in_name, s_target_qid=None):
        x, y = super(AttentionLes, self).pairwise_data_reader(in_name, s_target_qid)
        x = self._reshape_input(x)
        return x, y
    
    def pointwise_data_reader(self, in_name, s_target_qid=None):
        x, y = super(AttentionLes, self).pointwise_data_reader(in_name, s_target_qid)
        x = self._reshape_input(x)
        return x, y

    def _reshape_input(self, x):
        """
        reshape the input to meet sf_ground_shape, e_ground_shape, e_match_shape, ltr_shape
        the first dim of each array is the batch, will not be affected
        :param x:
        :return:
        """
        l_name_shape = zip([sf_ground_name, e_ground_name,
                            e_match_name, ltr_feature_name],
                           [self.sf_ground_shape, self.e_ground_shape,
                            self.e_match_shape, self.ltr_shape])
        for x_name, x_shape in l_name_shape:
            if x_name not in x:
                continue
            if x[x_name].shape[1:] != x_shape:
                x[x_name] = self._padding(x[x_name], x_shape)
                logging.info('reshape [%s] to shape %s', x_name, json.dumps(x_shape))

        l_aux_name_shape = [(self.aux_pre + name, shape) for name, shape in l_name_shape]
        for x_name, x_shape in l_aux_name_shape:
            if x_name not in x:
                continue
            if x[x_name].shape[1:] != x_shape:
                x[x_name] = self._padding(x[x_name], x_shape)
                logging.info('reshape [%s] to shape %s', x_name, json.dumps(x_shape))
        return x

    @classmethod
    def _padding(cls, ts, ts_shape):
        """
        reshape the 1: dim of ts
        only support 1-4 dim
        :param ts:
        :param ts_shape:
        :return:
        """
        nb_x = ts.shape[0]
        new_ts = np.zeros([nb_x] + list(ts_shape))

        if len(ts_shape) == 1:
            new_ts[:, :ts_shape[0]] = ts[:, :ts_shape[0]]
        if len(ts_shape) == 2:
            new_ts[:, :ts_shape[0], :ts_shape[1]] = ts[:, :ts_shape[0], :ts_shape[1]]
        if len(ts_shape) == 3:
            new_ts[:, :ts_shape[0], :ts_shape[1], :ts_shape[2]] = ts[:, :ts_shape[0], :ts_shape[1], :ts_shape[2]]
        if len(ts_shape) == 4:
            new_ts[:, :ts_shape[0], :ts_shape[1], :ts_shape[2], :ts_shape[3]] = ts[:, :ts_shape[0], :ts_shape[1], :ts_shape[2], :ts_shape[3]]

        return new_ts

    def _build_para_layers(self):
        """
        an sf grounding layer
        an entity grounding layer
        an ltr layer
        an entity matching layer
        :return:
        """
        # 1 d cnn to calculate the surface attention
        # attention model does not use bias to ensure padding
        sf_ground_cnn = Conv1D(
            nb_filter=1,
            filter_length=1,
            activation='relu',
            W_regularizer=l2(self.hyper_para.l2_w),
            bias=False,
            input_shape=self.sf_ground_shape,
            name=sf_ground_name + '_CNN',
        )

        # a typical ltr linear model
        ltr_dense = Dense(
            output_dim=1,
            bias=True,
            input_shape=self.ltr_shape,
            name=ltr_feature_name + '_Dense'
        )

        # 2 d cnn on each sf-e pair, size 1 means only apply a linear model on the feature dime
        #  will result a sf-e matrix, will then be soft-maxed for probability
        # attention model does not use bias to ensure padding
        e_ground_cnn = Conv2D(
            nb_filter=1,
            nb_row=1,
            nb_col=1,
            W_regularizer=l2(self.hyper_para.l2_w),
            bias=False,
            activation='linear',
            dim_ordering='tf',
            input_shape=self.e_ground_shape,
            name=e_ground_name + '_CNN'
        )

        # 2 d cnn to get matching scores for entities
        e_match_cnn = Conv2D(
            nb_filter=1,
            nb_row=1,
            nb_col=1,
            W_regularizer=l2(self.hyper_para.l2_w),
            bias=True,
            dim_ordering='tf',
            input_shape=self.e_match_shape,
            name=e_match_name + '_CNN'
        )

        h_para_layers = {
            sf_ground_name + '_CNN': sf_ground_cnn,
            e_ground_name + '_CNN': e_ground_cnn,
            ltr_feature_name + '_Dense': ltr_dense,
            e_match_name + '_CNN': e_match_cnn
        }
        return h_para_layers

    def _build_model(self):
        h_para_layers = self._build_para_layers()
        ranker = self._form_model_from_layers(h_para_layers, is_aux=False)
        aux_ranker = self._form_model_from_layers(h_para_layers, is_aux=True)

        training_model = Sequential()
        training_model.add(Merge([ranker, aux_ranker],
                                 mode=lambda x: x[0] - x[1],
                                 output_shape=(1,),
                                 name='training_pairwise'
                                 )
                           )
        logging.info('ranker summary')
        ranker.summary()
        logging.info('trainer summary')
        training_model.summary()
        self.ranking_model = ranker
        self.training_model = training_model
        return ranker, training_model

    def _form_model_from_layers(self, h_para_layers, is_aux=False):
        """
        merge sf_ground_cnn's 1d results with e_ground_cnn's
            sf_ground_cnn |sf| * 1, e_ground_cnn |sf||e|, multiply the vector along the cols
            to get a |sf||e| attention matrix
        then merge with e_match_cnn's results, a full dot to a single score
        then add with ltr's results to get the final ranking score
        :param h_para_layers: the returned results of _build_para_layers
        :return:
        """

        sf_ground_cnn = h_para_layers[sf_ground_name + '_CNN']
        e_ground_cnn = h_para_layers[e_ground_name + '_CNN']
        ltr_dense = h_para_layers[ltr_feature_name + '_Dense']
        e_match_cnn = h_para_layers[e_match_name + '_CNN']

        pre = ""
        if is_aux:
            pre = self.aux_pre

        # align inputs
        sf_ground_input = Input(shape=self.sf_ground_shape, name=pre + sf_ground_name)
        sf_ground_cnn = sf_ground_cnn(sf_ground_input)
        sf_ground_cnn = Flatten()(sf_ground_cnn)

        e_ground_input = Input(shape=self.e_ground_shape, name=pre + e_ground_name)
        e_ground_cnn = e_ground_cnn(e_ground_input)
        e_ground_cnn = Reshape(self.e_match_shape[:-1])(e_ground_cnn)  # drop last dimension
        if self.e_att_activation == 'softmax':
            e_ground_cnn = Activation('softmax')(e_ground_cnn)

        ltr_input = Input(shape=self.ltr_shape, name=pre + ltr_feature_name)
        ltr_dense = ltr_dense(ltr_input)

        e_match_input = Input(shape=self.e_match_shape, name=pre + e_match_name)
        e_match_cnn = e_match_cnn(e_match_input)
        e_match_cnn = Flatten()(e_match_cnn)

        # broad cast the sf's score to sf-e mtx
        sf_att = RepeatVector(self.max_e_per_spot)(sf_ground_cnn)
        sf_att = Permute((2, 1))(sf_att)

        e_combined_att = merge([sf_att, e_ground_cnn],
                               mode='mul', name=pre + 'full_att_mtx'
                               )

        e_ranking_score = merge([Flatten()(e_combined_att), e_match_cnn],
                                mode='dot', name=pre + 'att_e_ranking_score')

        ranking_score = merge([e_ranking_score, ltr_dense],
                              mode='sum', output_shape=(1,), name=pre+'ew_combine')
        ranking_model = Model(input=[sf_ground_input, e_ground_input, e_match_input, ltr_input],
                              output=ranking_score)

        return ranking_model


class Les(AttentionLes):
    model_name = Unicode('les')

    def _form_model_from_layers(self, h_para_layers, is_aux=False):
        """
        merge sf_ground_cnn's 1d results with e_ground_cnn's
            sf_ground_cnn |sf| * 1, e_ground_cnn |sf||e|, multiply the vector along the cols
            to get a |sf||e| attention matrix
        then merge with e_match_cnn's results, a full dot to a single score
        then add with ltr's results to get the final ranking score
        :param h_para_layers: the returned results of _build_para_layers
        :return:
        """

        ltr_dense = h_para_layers[ltr_feature_name + '_Dense']
        e_match_cnn = h_para_layers[e_match_name + '_CNN']

        pre = ""
        if is_aux:
            pre = self.aux_pre

        # align inputs

        ltr_input = Input(shape=self.ltr_shape, name=pre + ltr_feature_name)
        ltr_dense = ltr_dense(ltr_input)

        e_match_input = Input(shape=self.e_match_shape, name=pre + e_match_name)
        e_match_cnn = e_match_cnn(e_match_input)
        e_match_cnn = Reshape(self.e_match_shape[:-1])(e_match_cnn)  # drop last dimension
        # e_match_cnn = Lambda(lambda x: K.mean(x, axis=1), output_shape=(1, ))(e_match_cnn)
        e_match_cnn = AveragePooling1D(pool_length=self.e_match_shape[0])(e_match_cnn)
        e_match_cnn = Permute((2, 1))(e_match_cnn)
        e_match_cnn = AveragePooling1D(pool_length=self.e_match_shape[1])(e_match_cnn)
        e_match_cnn = Flatten()(e_match_cnn)
        # broad cast the sf's score to sf-e mtx

        #  use average pooling
        ranking_score = merge([e_match_cnn, ltr_dense],
                              mode='sum', output_shape=(1,), name=pre+'ew_combine')
        ranking_model = Model(input=[e_match_input, ltr_input],
                              output=ranking_score)

        return ranking_model


class SfAttLes(AttentionLes):
    """
    only apply attention on surface forms
    """
    model_name = Unicode('sf_les')
    max_e_per_spot = Int(1, help='top e allowed per q').tag(config=True)

    def _form_model_from_layers(self, h_para_layers, is_aux=False):
        """
        merge sf_ground_cnn's 1d results with e_ground_cnn's
            sf_ground_cnn |sf| * 1, e_ground_cnn |sf||e|, multiply the vector along the cols
            to get a |sf||e| attention matrix
        then merge with e_match_cnn's results, a full dot to a single score
        then add with ltr's results to get the final ranking score
        :param h_para_layers: the returned results of _build_para_layers
        :return:
        """

        sf_ground_cnn = h_para_layers[sf_ground_name + '_CNN']
        ltr_dense = h_para_layers[ltr_feature_name + '_Dense']
        e_match_cnn = h_para_layers[e_match_name + '_CNN']

        pre = ""
        if is_aux:
            pre = self.aux_pre

        # align inputs
        sf_ground_input = Input(shape=self.sf_ground_shape, name=pre + sf_ground_name)
        sf_ground_cnn = sf_ground_cnn(sf_ground_input)
        sf_ground_cnn = Flatten()(sf_ground_cnn)

        ltr_input = Input(shape=self.ltr_shape, name=pre + ltr_feature_name)
        ltr_dense = ltr_dense(ltr_input)

        e_match_input = Input(shape=self.e_match_shape, name=pre + e_match_name)
        e_match_cnn = e_match_cnn(e_match_input)
        e_match_cnn = Flatten()(e_match_cnn)

        # broad cast the sf's score to sf-e mtx
        # TODO this must be the problem. Check WHY
        sf_att = RepeatVector(self.max_e_per_spot)(sf_ground_cnn)
        sf_att = Permute((2, 1), name='sf_att')(sf_att)

        e_ranking_score = merge([Flatten()(sf_att), e_match_cnn],
                                mode='dot', name=pre + 'att_e_ranking_score')

        ranking_score = merge([e_ranking_score, ltr_dense],
                              mode='sum', output_shape=(1,), name=pre+'ew_combine')
        ranking_model = Model(input=[sf_ground_input, e_match_input, ltr_input],
                              output=ranking_score)

        return ranking_model

    def predict(self, x):
        """
        add a log about the intermediate results of sf_att
        :param x:
        :return:
        """

        # get sf att intermediate results, and put to log (per qid's sf att mtx)
        logging.info('fetching intermediate results')
        name = sf_ground_name + '_CNN'
        layer = self.ranking_model.get_layer(name)
        intermediate_model = Model(input=layer.get_input_at(0),
                                   output=layer.get_output_at(0)
                                   )
        intermediate_model.summary()
        mid_res = intermediate_model.predict(x)
        l_meta = x['meta']
        s_qid = {}
        for p in xrange(mid_res.shape[0]):
            qid = l_meta[p]['qid']
            if qid not in s_qid:
                s_qid[qid] = True
                logging.info('sf_att of q [%s]: %s', qid, np.array2string(mid_res[p]))
        y = super(SfAttLes, self).predict(x)
        return y


class DisAmbiAttLes(AttentionLes):
    """
    only apply attention on entity disambiguation
    """
    model_name = Unicode('dis_ambi_les')

    def _form_model_from_layers(self, h_para_layers, is_aux=False):
        """
        merge sf_ground_cnn's 1d results with e_ground_cnn's
            sf_ground_cnn |sf| * 1, e_ground_cnn |sf||e|, multiply the vector along the cols
            to get a |sf||e| attention matrix
        then merge with e_match_cnn's results, a full dot to a single score
        then add with ltr's results to get the final ranking score
        :param h_para_layers: the returned results of _build_para_layers
        :return:
        """

        e_ground_cnn = h_para_layers[e_ground_name + '_CNN']
        ltr_dense = h_para_layers[ltr_feature_name + '_Dense']
        e_match_cnn = h_para_layers[e_match_name + '_CNN']

        pre = ""
        if is_aux:
            pre = self.aux_pre

        # align inputs

        e_ground_input = Input(shape=self.e_ground_shape, name=pre + e_ground_name)
        e_ground_cnn = e_ground_cnn(e_ground_input)
        e_ground_cnn = Reshape(self.e_match_shape[:-1])(e_ground_cnn)  # drop last dimension
        if self.e_att_activation == 'softmax':
            e_ground_cnn = Activation('softmax')(e_ground_cnn)

        ltr_input = Input(shape=self.ltr_shape, name=pre + ltr_feature_name)
        ltr_dense = ltr_dense(ltr_input)

        e_match_input = Input(shape=self.e_match_shape, name=pre + e_match_name)
        e_match_cnn = e_match_cnn(e_match_input)
        e_match_cnn = Flatten()(e_match_cnn)

        # broad cast the sf's score to sf-e mtx

        e_ranking_score = merge([Flatten()(e_ground_cnn), e_match_cnn],
                                mode='dot', name=pre + 'att_e_ranking_score')

        ranking_score = merge([e_ranking_score, ltr_dense],
                              mode='sum', output_shape=(1,), name=pre+'ew_combine')
        ranking_model = Model(input=[e_ground_input, e_match_input, ltr_input],
                              output=ranking_score)

        return ranking_model
