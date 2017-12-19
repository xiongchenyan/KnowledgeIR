"""

"""

import keras
from keras.layers import (
    Merge,
    Input,
    Flatten,
    Convolution1D,
)
from keras.models import (
    Model,
    Sequential,
)
# from keras.layers.pooling import GlobalAveragePooling1D
from keras.regularizers import (
    l2,
)
from traitlets import (
    Int,
    Unicode,
)

from knowledge4ir.duet_model import AttLeToR


class HierarchicalAttLeToR(AttLeToR):
    nb_middle_filters = Int(5).tag(config=True)
    activation = Unicode('linear').tag(config=True)
    att_activation = Unicode('linear').tag(config=True)

    def _build_model(self):
        l_inputs = self._init_inputs()
        l_aux_inputs = self._init_inputs(is_aux=True)
        l_models = self._init_model()

        att_ranker = self._align_to_rank_model(l_inputs, l_models)
        aux_att_ranker = self._align_to_rank_model(l_aux_inputs, l_models)
        pairwise_trainer = Sequential()
        pairwise_trainer.add(Merge([att_ranker, aux_att_ranker],
                                   mode=lambda x: x[0] - x[1],
                                   output_shape=(1,)
                                   ))
        return att_ranker, pairwise_trainer

    def _init_inputs(self, is_aux=False):
        l_inputs = []
        for name, dim in zip(self.l_model_names, self.l_input_dim):
            if is_aux:
                in_name = self.aux_pre + name
            else:
                in_name = name
            input_layer = Input(shape=dim,
                                name=in_name)
            l_inputs.append(input_layer)

        return l_inputs

    def _init_model(self):
        l_models = []
        l_in_shape = self.l_input_dim
        l_model_name = [name + '_model' for name in self.l_model_names]
        l_nb_layer = [self.nb_rank_layer, self.nb_rank_layer, self.nb_att_layer, self.nb_att_layer]
        for p in xrange(len(l_in_shape)):
            if p < 2:
                activation = self.activation
                l2 = self.l2_w
            else:
                activation = self.att_activation
                l2 = self.att_l2_w
            model = self._init_one_neural_network(
                l_in_shape[p],
                l_model_name[p],
                l_nb_layer[p],
                activation,
                l2
            )
            l_models.append(model)
        return l_models

    def _align_to_rank_model(self, l_inputs, l_models):
        l_aligned_models = [model(input) for model, input in zip(l_models, l_inputs)]
        # ranker_model = Merge(mode='concat', name='rank_merge')(l_aligned_models[:2])
        ranker_model = keras.layers.Concatenate(name='rank_cat')(l_aligned_models[:2])
        # att_model = Merge(mode='concat', name='att_merge')(l_aligned_models[2:])
        att_model = keras.layers.Concatenate(name='att_cat')(l_aligned_models[2:])
        # att_ranker = Merge(mode='dot', dot_axes=-1,name='att_rank_dot_merge')([ranker_model, att_model])
        att_ranker = keras.layers.Dot(axes=-1, name='att_rank_dot')([ranker_model, att_model])
        att_ranker = Model(input=l_inputs, output=att_ranker)
        return att_ranker

    def _init_one_neural_network(self, in_shape, model_name, nb_layer, activation='linear', l2_w=None):
        model = Sequential(name=model_name)
        if not l2_w:
            l2_w = self.l2_w
        for lvl in xrange(nb_layer):
            if lvl == nb_layer - 1:
                this_nb_filter = 1
            else:
                this_nb_filter = self.nb_middle_filters
            if lvl == 0:
                this_layer = Convolution1D(nb_filter=this_nb_filter,
                                           filter_length=1,
                                           input_shape=in_shape,
                                           activation=activation,
                                           bias=False,
                                           W_regularizer=l2(l2_w)
                                           )

            else:
                this_layer = Convolution1D(nb_filter=this_nb_filter,
                                           filter_length=1,
                                           activation=activation,
                                           bias=False,
                                           W_regularizer=l2(l2_w)
                                           )
            model.add(this_layer)
        model.add(Flatten())
        return model


