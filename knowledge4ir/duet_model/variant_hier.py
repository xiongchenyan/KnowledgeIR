from keras.engine import Model
from keras.layers import TimeDistributed, Dense, Flatten, Masking, Activation
from keras.legacy.layers import Merge
from keras.models import Sequential
from keras.regularizers import l2

from knowledge4ir.duet_model import HierarchicalAttLeToR


class MaskHierarchicalAttLeToR(HierarchicalAttLeToR):
    def _align_to_rank_model(self, l_inputs, l_models):
        l_aligned_models = [model(input) for model, input in zip(l_models, l_inputs)]
        ranker_model = Merge(mode='concat', name='rank_merge')(l_aligned_models[:2])
        att_model = Merge(mode='concat', name='att_merge')(l_aligned_models[2:])
        att_ranker = Merge(mode='dot',
                           name='att_rank_dot_merge'
                           )([ranker_model, att_model])
        att_ranker = Model(input=l_inputs, output=att_ranker)
        return att_ranker

    def _init_one_neural_network(self, in_shape, model_name, nb_layer, activation='linear'):
        model = Sequential(name=model_name)
        this_nb_filter = self.nb_middle_filters

        # model.add(Masking(input_shape=in_shape, mask_value=0.))

        for lvl in xrange(nb_layer):
            if lvl == nb_layer - 1:
                this_nb_filter = 1
            if lvl == 0:
                this_layer = TimeDistributed(
                    Dense(this_nb_filter,
                          activation=self.activation,
                          bias=False,
                          W_regularizer=l2(self.l2_w)
                          ),
                    input_shape=in_shape
                )
            else:
                this_layer = TimeDistributed(
                    Dense(this_nb_filter,
                          activation=self.activation,
                          bias=False,
                          W_regularizer=l2(self.l2_w)
                          ),
                )
            model.add(this_layer)
        model.add(Flatten())
        return model


class ProbAttLeToR(HierarchicalAttLeToR):
    def _align_to_rank_model(self, l_inputs, l_models):
        l_aligned_models = [model(input) for model, input in zip(l_models, l_inputs)]
        ranker_model = Merge(mode='concat', name='rank_merge')(l_aligned_models[:2])
        ranker_model = Masking()(ranker_model)
        att_model = Merge(mode='concat', name='att_merge')(l_aligned_models[2:])
        att_model = Masking()(att_model)
        att_model = Activation('softmax', name='softmax_attention')(att_model)
        att_ranker = Merge(mode='dot', dot_axes=-1,name='att_rank_dot_merge'
                           )([ranker_model, att_model])
        att_ranker = Model(input=l_inputs, output=att_ranker)
        return att_ranker