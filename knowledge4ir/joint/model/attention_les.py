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
    Lambda,
    Reshape,
    Input,
    RepeatVector,
    Lambda,
)
from keras.activations import (
    softmax,
    relu,
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
    JointSemanticModel)
from knowledge4ir.joint import (
    JointSemanticModel
)
from traitlets import (
    Unicode,
    Int,
    List,
)


class AttentionLes(JointSemanticModel):
    model_name = Unicode('att_les')
    max_spot_per_q = Int(3, help='max spot allowed per q').tag(config=True)
    max_e_per_spot = Int(5, help='top e allowed per q').tag(config=True)
    sf_ground_f_dim = Int(6, help='sf ground feature dimension').tag(config=True)
    e_ground_f_dim = Int(5, help='e ground feature dimension').tag(config=True)
    e_match_f_dim = Int(15, help='e match feature dimension').tag(config=True)
    ltr_f_dim = Int(1, help='ltr feature dimension').tag(config=True)
    l_x_name = List(Unicode, default_value=l_input_name).tag(config=True)

    def __init__(self, **kwargs):
        super(AttentionLes, self).__init__(**kwargs)
        self.sf_ground_shape = (self.max_spot_per_q, self.sf_ground_f_dim)
        self.e_ground_shape = (self.max_spot_per_q, self.max_e_per_spot, self.e_ground_f_dim)
        self.e_match_shape = (self.max_spot_per_q, self.max_e_per_spot, self.e_ground_f_dim)
        self.ltr_shape = (self.ltr_f_dim,)

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
            activation='softmax',
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
                                 )
                           )
        logging.info('ranker summary: %s', ranker.to_json(indent=1))
        ranker.summary()
        logging.info('trainer summary: %s', training_model.to_json(indent=1))
        training_model.summary()
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
        e_ground_cnn = h_para_layers[e_ground_name + 'CNN']
        ltr_dense = h_para_layers[ltr_feature_name + 'Dense']
        e_match_cnn = h_para_layers[e_match_name + '_CNN']

        pre = ""
        if is_aux:
            pre = self.aux_pre

        # align inputs
        sf_ground_input = Input(shape=self.sf_ground_shape,
                                name=pre + sf_ground_name)
        sf_ground_cnn = sf_ground_cnn(sf_ground_input)

        e_ground_input = Input(shape=self.e_ground_shape,
                               name=pre + e_ground_name)
        e_ground_cnn = e_ground_cnn(e_ground_input)

        ltr_input = Input(shape=self.ltr_shape,
                          name=pre + ltr_feature_name)
        ltr_dense = ltr_dense(ltr_input)

        e_match_input = Input(shape=self.e_match_shape,
                              name=pre + e_match_name)
        e_match_cnn = e_match_cnn(e_match_input)

        # broad cast the sf's score to sf-e mtx
        sf_att = RepeatVector(self.max_e_per_spot)(sf_ground_cnn)
        sf_att = Lambda(lambda x: K.transpose(x))(sf_att)

        e_combined_att = merge([sf_att, e_ground_cnn],
                               mode='mul')

        e_ranking_score = merge([Flatten()(e_combined_att), Flatten()(e_match_cnn)],
                                mode='dot',
                                )

        ranking_score = merge([e_ranking_score, ltr_dense],
                              mode='sum')
        ranking_model = Model(input=[sf_ground_input, e_ground_input, e_match_input, ltr_input],
                              output=ranking_score)

        return ranking_model



