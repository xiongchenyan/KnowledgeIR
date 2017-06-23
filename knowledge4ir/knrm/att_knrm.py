"""
knrm with attention
"""
import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense, concatenate, Reshape, multiply, Conv1D
from keras.legacy.layers import Merge
from keras.models import Sequential
from traitlets import Bool, Int, Unicode

from knowledge4ir.knrm.distance_metric import DiagnalMetric
from knowledge4ir.knrm.kernel_pooling import KernelPooling, KpLogSum
from knowledge4ir.knrm.model import KNRM
from knowledge4ir.knrm import q_len, l_field_len
import logging


class AttKNRM(KNRM):
    """
    attention version of KNRM
    will directly take input of the calculated translation matrix
        q-d field matrix
    can config whether to use attention or not
    q att and d att is to be multiplied to the kernel pooled raw score tensors,
        alone corresponding dimension (q:1, d:2)
    attention mechanism is a dense layer with input features for now (06/22/2017)
    """
    translation_mtx_in = 'translation_mtx'
    with_attention = Bool(True, help='whether to use attention').tag(config=True)
    att_dim = Int(7, help='attention feature dimension').tag(config=True)

    # overide not in use configs
    embedding_dim = Int()
    vocab_size = Int()
    metric_learning = Unicode()

    def __init__(self, **kwargs):
        super(AttKNRM, self).__init__(**kwargs)
        l_in_names = [self.ltr_feature_name] + [self.translation_mtx_in + '_' + self.d_name + '_' + field for field in self.l_d_field]
        l_in_names += [self.aux_pre + name for name in l_in_names]
        self.s_target_inputs = set(
            l_in_names + ['qid', 'docno', 'docno_pair', 'y']
        )
        if self.with_attention:
            l_att_names = [self.d_att_name + '_' + field for field in self.l_d_field]
            l_att_names = [self.aux_pre + name for name in l_att_names]
            self.s_target_inputs |= set([self.q_att_name] + l_att_names)
        self.l_d_layer = []
        self.q_len = q_len
        self.l_field_len = l_field_len

    def set_embedding(self, pretrained_emb):
        logging.warn('att knrm does not use embedding')
        pass

    def _init_inputs(self):
        l_field_translation = self._init_translation_input()
        l_aux_field_translation = self._init_translation_input(aux=True)

        ltr_input, aux_ltr_input = None, None
        if self.ltr_feature_dim > 0:
            ltr_input = Input(shape=(self.ltr_feature_dim,),
                              name=self.ltr_feature_name)
            aux_ltr_input = Input(shape=(self.ltr_feature_dim,),
                                  name=self.aux_pre + self.ltr_feature_name)
        q_att_input = None
        l_field_att_input = []
        l_aux_field_att_input = []
        if self.with_attention:
            q_att_input, l_field_att_input = self._init_att_input()
            __, l_aux_field_att_input = self._init_att_input(aux=True)
        l_inputs = [
            l_field_translation, l_aux_field_translation,
            ltr_input, aux_ltr_input,
            q_att_input, l_field_att_input, l_aux_field_att_input
        ]
        return l_inputs

    def _init_att_input(self, aux=False):
        pre = ""
        q_att_input = None
        if aux:
            pre = self.aux_pre
        else:
            q_att_input = Input(shape=(self.q_len, self.att_dim,), name=self.q_att_name)
        l_field_att_input = [Input(shape=(f_len, self.att_dim,), name=pre + self.d_att_name + '_' + field)
                             for field, f_len in zip(self.l_d_field, self.l_field_len)
                             ]
        return q_att_input, l_field_att_input

    def _init_translation_input(self, aux=False):
        pre = ""
        if aux:
            pre = self.aux_pre
        l_field_translation = []
        for field in self.l_d_field:
            l_field_translation.append(
                Input(shape=(None, None,),
                      name=pre + self.translation_mtx_in + '_' + self.d_name + '_' + field,
                      dtype='float32')
            )
        return l_field_translation

    def _init_layers(self):
        self.kernel_pool = KernelPooling(
            np.array(self.mu), np.array(self.sigma), use_raw=True, name='kp')
        self.kp_logsum = KpLogSum(name='kp_logsum')
        self.ltr_layer = Dense(
            1,
            name='letor',
            use_bias=False,
            input_dim=len(self.l_d_field) * len(self.mu) + self.ltr_feature_dim
        )
        if self.with_attention:
            self.q_att = Conv1D(
                filters=1,
                kernel_size=1,
                use_bias=False,
                input_shape=(self.q_len, self.att_dim),
                name='dense_q_att'
            )
            self.l_field_att = [
                Conv1D(
                    filters=1,
                    kernel_size=1,
                    use_bias=False,
                    input_shape=(f_len, self.att_dim),
                    name='dense_d_%s_att' % field
                    )
                for field, f_len in zip(self.l_d_field, self.l_field_len)
                ]

    def _init_translation_ranker(self, l_field_translate, ltr_input=None,
                                 q_att_input=None, l_field_att_input=None,
                                 aux=False):
        """
        construct ranker for given inputs
        :param l_field_translate: translaiton matrices
        :param ltr_input: if use ltr features to combine
        :param q_att_input: q attention input
        :param l_field_att_input: field attention input
        :param aux:
        :return:
        """
        pre = ""
        if aux:
            pre = self.aux_pre
        q_att = None
        l_field_att = []
        if self.with_attention:
            if not aux:
                self.l_q_att_in = q_att_input
                self.l_field_att_in = l_field_att_input
            q_att = Reshape(target_shape=(-1, 1, 1))(self.q_att(q_att_input))
            l_field_att = [
                Reshape(target_shape=(1, -1, 1))(self.l_field_att[p](l_field_att_input[p]))
                for p in xrange(len(self.l_field_att))
                ]

        # perform kernel pooling
        l_kp_features = []
        for p in xrange(len(self.l_d_field)):
            # field = self.l_d_field[p]
            f_in = l_field_translate[p]
            d_layer = self.kernel_pool(f_in)
            # TODO test
            if self.with_attention:
                # need custom multiple layer to do * along target axes
                # use broadcast reshape attention to targeted dimensions, and then use multiply
                # q_att = Reshape(target_shape=(-1, 1, 1))(q_att)
                d_layer = multiply([d_layer, q_att])
                # l_field_att[p] = Reshape(target_shape=(1, -1, 1))(l_field_att[p])
                d_layer = multiply([d_layer, l_field_att[p]])
            if not aux:
                self.l_d_layer.append(d_layer)
            d_layer = self.kp_logsum(d_layer)
            l_kp_features.append(d_layer)

        # put features to one vector
        if len(l_kp_features) > 1:
            ranking_features = concatenate(l_kp_features, name=pre + 'ranking_features')
        else:
            ranking_features = l_kp_features[0]

        # # test
        # test_model = Model(inputs=l_field_translate, outputs=ranking_features)
        # test_model.summary()

        if ltr_input:
            ranking_features = concatenate([ranking_features, ltr_input],
                                           name=pre + 'ranking_features_with_ltr')

        ranking_layer = self.ltr_layer(ranking_features)
        l_full_inputs = l_field_translate
        if self.with_attention:
            l_full_inputs.append(q_att_input)
            l_full_inputs.extend(l_field_att_input)
        if ltr_input:
            l_full_inputs.append(ltr_input)
        ranker = Model(inputs=l_full_inputs,
                       outputs=ranking_layer,
                       name=pre + 'ranker')

        return ranker

    def construct_model_via_translation(
            self, l_field_translation, l_aux_field_translation, ltr_input, aux_ltr_input,
            q_att_input, l_field_att_input, l_aux_field_att_input
    ):
        ranker = self._init_translation_ranker(l_field_translation, ltr_input,
                                               q_att_input, l_field_att_input
                                               )
        aux_ranker = self._init_translation_ranker(l_aux_field_translation, aux_ltr_input,
                                                   q_att_input, l_aux_field_att_input,
                                                   True)
        trainer = Sequential()
        trainer.add(
            Merge([ranker, aux_ranker],
                  mode=lambda x: x[0] - x[1],
                  output_shape=(1,),
                  name='training_pairwise'
                  )
        )
        return ranker, trainer

    def build(self):
        l_inputs = self._init_inputs()
        self.l_field_translation = l_inputs[0]
        l_field_translation, l_aux_field_translation = l_inputs[:2]
        ltr_input, aux_ltr_input = l_inputs[2:4]
        q_att_input, l_field_att_input, l_aux_field_att_input = l_inputs[4:]
        self._init_layers()
        self.ranker, self.trainer = self.construct_model_via_translation(
            l_field_translation, l_aux_field_translation, ltr_input, aux_ltr_input,
            q_att_input, l_field_att_input, l_aux_field_att_input
        )
        return self.ranker, self.trainer


if __name__ == '__main__':
    """
    unit testing
    0) compile and check the model parameters: pass
    1) whether the translation matrices are correct: pass
    2) whether the kernel pooling is right: pass
    3) whether kp_logsum is correct: pass
    4) whether the attention works
    """
    import numpy as np
    # 0)
    att_knrm = AttKNRM()
    att_knrm.build()
    att_knrm.ranker.summary()
    att_knrm.trainer.summary()

    ll = [[
        [1, 0.5, 0.9, 0, 0.3],
        [0.5, 0, 0, 0, 1]
    ]]
    q_att = np.array([[[1, 1] + [0] * 5, [0] * 7]])
    d_att = np.zeros((1, 5, 7))
    d_att[0:3, 0] = 1
    # 1)
    trans_mtx = np.array(ll)
    print trans_mtx

    # 2) + 4)
    kp = Model(inputs=[att_knrm.l_field_translation[0],
                       att_knrm.l_q_att_in,
                       att_knrm.l_field_att_in[0]],
               outputs=att_knrm.l_d_layer[0])
    kp_res = kp.predict([trans_mtx, q_att, d_att])
    print 'raw kernel scores with attention'
    print kp_res.shape
    print kp_res

    # # 3)
    # kp = Model(inputs=att_knrm.l_field_translation[0],
    #            outputs=att_knrm.kp_logsum(att_knrm.l_d_layer[0]))
    # kp_res = kp.predict(trans_mtx)
    # print 'log summed kernel features'
    # print kp_res.shape
    # print kp_res

