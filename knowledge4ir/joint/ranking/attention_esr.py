"""
soft ESR model
takes the soft-entities sequence of q and d as input
uses embedding as the external resource
uses hyperparameter class input
"""

from keras.layers import (
    merge,
    Merge,
    Lambda,
    Dense,
    Input,
    Embedding,
    Reshape,
)
from keras.regularizers import (
    l2,
)
from keras.models import (
    Model,
    Sequential,
)
from knowledge4ir.joint import (
    JointSemanticModel,
    JointSemanticResource,
)

from traitlets.config import Configurable
from traitlets import (
    Float,
    Int,
    Tuple,
    Unicode
)
from knowledge4ir.joint.kernel_pooling import kernel_pooling


# TODO: wrong!
# TODO: The attention should be applied on the k-p's score, but not on the embedding nor the translation score
# TODO: On hold 03/08/2017
class AttentionESR(JointSemanticModel):

    def _build_para_layers(self):
        h_layer = dict()

        h_layer['rank_nn'] = Dense(
            output_dim=1,
            W_regularizer=l2(self.hyper_para.l2_w)
        )

        return

    def _form_model_from_layers(self, h_para_layers):
        ranking_model = self._form_ranking_model(h_para_layers, prefix='')
        aux_ranking_model = self._form_ranking_model(h_para_layers, prefix=self.aux_pre)
        training_model = Sequential()
        training_model.add(Merge([ranking_model, aux_ranking_model],
                           mode=lambda x: x[0] - x[1],
                                 )
                           )
        return ranking_model, training_model

    def _form_ranking_model(self, h_para_layers, prefix=""):

        q_shape, title_shape, body_shape = self.hyper_para.q_shape, self.hyper_para.title_shape, self.hyper_para.body_shape

        q_emb, q_in, q_att_in = self._form_attention_emb(
            q_shape,
            self.q_name,
            self.q_att,
            prefix
        )
        t_emb, t_in, t_att_in = self._form_attention_emb(
            title_shape,
            self.title_name,
            self.title_att,
            prefix
        )
        d_emb, d_in, d_att_in = self._form_attention_emb(
            body_shape,
            self.body_name,
            self.body_att,
            prefix
        )

        qt_trans_mtx = merge([q_emb, t_emb],
                             mode='dot',
                             dot_axes=(1, 1)
                             )  # TODO test
        qd_trans_mtx = merge([q_emb, d_emb],
                             mode='dot',
                             dot_axes=(1, 1)
                             )

        h_pooling_para = dict()
        h_pooling_para['l_mean'] = self.hyper_para.l_kernel_pool_mean
        h_pooling_para['l_sigma'] = [self.hyper_para.l_kernel_pool_sigma]

        qt_kernel_pool = Lambda(kernel_pooling,
                                output_shape=len(self.hyper_para.l_kernel_pool_mean,),
                                arguments=h_pooling_para
                                )(qt_trans_mtx)
        qd_kernel_pool = Lambda(kernel_pooling,
                                output_shape=len(self.hyper_para.l_kernel_pool_mean,),
                                arguments=h_pooling_para
                                )(qd_trans_mtx)

        kp_feature = merge([qt_kernel_pool, qd_kernel_pool],
                           mode='cat')

        ranker = h_para_layers['rank_nn'](kp_feature)

        ranking_model = Model(
            input=[q_in, q_att_in, t_in, t_att_in, d_in, d_att_in],
            output=ranker
        )
        return ranking_model

    def _form_attention_emb(self, data_shape, data_name, att_name, prefix=""):
        data_in = Input(shape=data_shape,
                        name=prefix + data_name
                        )
        data_in = Reshape((data_shape[0] * data_shape[1], )
                          )(data_in)

        data_emb_layer = Embedding(
            self.external_resource.embedding.shape[0],
            self.external_resource.embedding.shape[1],
            weights=[self.external_resource.embedding],
            trainable=False,
            input_length=data_shape[0] * data_shape[1]
        )(data_in)

        data_att_in = Input(shape=data_shape,
                         name=prefix + att_name,
                         )

        data_emb_layer = Reshape(
            (data_shape[0], data_shape[1], self.external_resource.embedding.shape[1])
        )(data_emb_layer)

        data_emb = merge([data_att_in, data_emb_layer],
                         dot_axes=[(1, ), (1, 2)]
                         )  # TODO test

        return data_emb, data_in, data_att_in

