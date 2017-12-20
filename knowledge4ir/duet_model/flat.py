from keras import Input, backend as K
from keras.engine import Model
from keras.layers import Convolution1D, Lambda
from keras.legacy.layers import Merge
from keras.models import Sequential
from keras.regularizers import l2
from traitlets import Int

from knowledge4ir.duet_model import HierarchicalAttLeToR


class FlatLeToR(HierarchicalAttLeToR):
    model_st = Int(0, help='start of the ranking model, 0 is qt ranker, 1 is qe ranker'
                   ).tag(config=True)
    model_ed = Int(2, help='end of the ranking model').tag(config=True)

    def _init_inputs(self, is_aux=False):
        l_inputs = []
        for name, dim in zip(self.l_model_names[self.model_st:self.model_ed],
                             self.l_input_dim[self.model_st:self.model_ed]):
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
        l_in_shape = self.l_input_dim[self.model_st:self.model_ed]
        l_model_name = [name + '_model'
                        for name in self.l_model_names[self.model_st:self.model_ed]]
        l_nb_layer = [self.nb_rank_layer, self.nb_rank_layer]
        for p in xrange(len(l_in_shape)):
            model = self._init_one_neural_network(
                l_in_shape[p],
                l_model_name[p],
                l_nb_layer[p]
            )
            l_models.append(model)
        return l_models

    def predict_intermediate(self, test_lines):
        raise NotImplementedError

    def _init_one_neural_network(self, in_shape, model_name, nb_layer,):
        model = Sequential(name=model_name)
        # model.add(Masking(input_shape=in_shape, mask_value=0.))
        this_nb_filter = self.nb_middle_filters
        for lvl in xrange(nb_layer):
            if lvl == nb_layer - 1:
                this_nb_filter = 1
            if lvl == 0:
                this_layer = Convolution1D(nb_filter=this_nb_filter,
                                           filter_length=1,
                                           input_shape=in_shape,
                                           activation=self.activation,
                                           bias=False,
                                           W_regularizer=l2(self.l2_w)
                                           )
                # this_layer = TimeDistributed(
                #     Dense(this_nb_filter,
                #           activation=self.activation,
                #           bias=False,
                #           W_regularizer=l2(self.l2_w)
                #           ),
                #     input_shape=in_shape,
                # )

            else:
                this_layer = Convolution1D(nb_filter=this_nb_filter,
                                           filter_length=1,
                                           activation=self.activation,
                                           bias=False,
                                           W_regularizer=l2(self.l2_w)
                                           )
                # this_layer = TimeDistributed(
                #     Dense(this_nb_filter,
                #           activation=self.activation,
                #           bias=False,
                #           W_regularizer=l2(self.l2_w)
                #           )
                # )
            model.add(this_layer)

        # model.add(Flatten())
        avg = Lambda(lambda x: K.mean(x, axis=1),
                     output_shape=(1,),
                     )
        # avg.supports_masking = True
        model.add(avg)
        # model.add(GlobalAveragePooling1D())
        return model

    def _align_to_rank_model(self, l_inputs, l_models):
        l_aligned_models = [model(this_input)
                            for model, this_input in zip(l_models, l_inputs)]
        if len(l_aligned_models) > 1:
            ranker_model = Merge(mode='sum', name='rank_merge')(l_aligned_models)
        else:
            ranker_model = l_aligned_models[0]
        # ranker_model = Lambda(lambda x: K.mean(x, axis=None),
        #                       output_shape=(1,)
        #                       )(ranker_model)
        # ranker_model = Dense(output_dim=1)(ranker_model)
        att_ranker = Model(input=l_inputs, output=ranker_model)
        return att_ranker


class QTermLeToR(HierarchicalAttLeToR):
    model_st = Int(0)
    model_ed = Int(1)

    # def _align_to_rank_model(self, l_inputs, l_models):
    #     l_aligned_models = [Lambda(lambda x: K.mean(x, axis=None),
    #                                output_shape=(1,)
    #                                )(model(spot)) for model, spot in zip(l_models, l_inputs)]
    #     ranker_model = l_aligned_models[0]
    #     # ranker_model = Lambda(lambda x: K.mean(x, axis=None),
    #     #                       output_shape=(1,)
    #     #                       )(ranker_model)
    #     # ranker_model = Dense(output_dim=1)(ranker_model)
    #     att_ranker = Model(spot=l_inputs, output=ranker_model)
    #     return att_ranker


class QEntityLeToR(QTermLeToR):
    model_st = Int(1)
    model_ed = Int(2)

    # def _align_to_rank_model(self, l_inputs, l_models):
    #     l_aligned_models = [model(spot) for model, spot in zip(l_models, l_inputs)]
    #     ranker_model = Lambda(lambda x: K.mean(x),
    #                           output_shape=(1,)
    #                           )(l_aligned_models[1])
    #     att_ranker = Model(spot=l_inputs, output=ranker_model)
    #     return att_ranker