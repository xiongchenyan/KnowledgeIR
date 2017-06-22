"""
knrm model
create knrm model
    input:
        q_boe
        d_boe
        d_att
        if train:
            aux_d_boe
            aux_d_att
    embedding use fixed parameter

"""

from keras.layers import (
    Embedding,
    Dense,
    Input,
    add,
    dot,
    Multiply,
    Lambda,
    Reshape,
    concatenate,
    Concatenate,
    Merge,
    LSTM,
)
from keras.activations import tanh, softmax
from keras.models import (
    Sequential,
    Model
)
from knowledge4ir.knrm.kernel_pooling import KernelPooling
from knowledge4ir.knrm.distance_metric import DiagnalMetric
import numpy as np
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode,
    List,
    Float,
    Bool,
)
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
)
from knowledge4ir.knrm import (
    aux_pre,
    q_in_name,
    d_in_name,
    ltr_feature_name,
    q_att_name,
    d_att_name
)
"""TODO
WARNING, the padding is not masked.
Now there will be 0 translation matrix elements counted by the kernels.
When using attention, since padding's attention feature is always 0, this will be addressed.
Keras's dot merge seems not propagating masking correctly.
"""


class KNRM(Configurable):
    """
    init knrm model
    ranking and training together
    """
    embedding_dim = Int(50, help='embedding dim').tag(config=True)
    vocab_size = Int(help='vocab size').tag(config=True)
    metric_learning = Bool(False, help='whether to learn the distance metric upon embedding'
                           ).tag(config=True)
    q_name = q_in_name
    d_name = d_in_name
    d_att_name = d_att_name
    q_att_name = q_att_name
    ltr_feature_name = ltr_feature_name
    aux_pre = aux_pre

    ltr_feature_dim = Int(1, help='ltr feature dimension, if 0 then no feature').tag(config=True)
    l_d_field = List(Unicode, default_value=TARGET_TEXT_FIELDS,
                     help='fields in the documents').tag(config=True)
    # q_len = Int(5, help='maximum q entity length')
    # l_d_field_len = List(Int, default_value=[10, 500],
    #                      help='max len of each field').tag(config=True)
    mu = List(Float,
              default_value=[1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9],
              help='mu of kernel pooling'
              ).tag(config=True)
    sigma = List(Float,
                 default_value=[1e-3] + [0.1] * 10,
                 help='sigma of kernel pooling').tag(config=True)

    def __init__(self, **kwargs):
        super(KNRM, self).__init__(**kwargs)
        self.emb = None

        self.q_input = None
        self.l_field_input = []
        self.l_aux_field_input = []

        self.ranking_layer = None
        self.ranker = None
        self.trainer = None

    def set_embedding(self, pretrained_emb):
        self.emb = pretrained_emb
        self.vocab_size, self.embedding_dim = self.emb.shape

    def _init_inputs(self):
        q_input, l_field_input = self._init_boe_input()
        __, l_aux_field_input = self._init_boe_input(aux=True)
        self.q_input = q_input
        self.l_field_input = l_field_input
        self.l_aux_field_input = l_aux_field_input
        if self.ltr_feature_dim > 0:
            ltr_input = Input(shape=(self.ltr_feature_dim,),
                              name=self.ltr_feature_name)
            aux_ltr_input = Input(shape=(self.ltr_feature_dim,),
                                  name=self.aux_pre + self.ltr_feature_name)
            return q_input, l_field_input, l_aux_field_input, ltr_input, aux_ltr_input
        else:
            return q_input, l_field_input, l_aux_field_input, None, None

    def _init_boe_input(self, aux=False):
        pre = ""
        if aux:
            pre = self.aux_pre
        q_input = Input(shape=(None,), name=pre + self.q_name, dtype='int32')
        l_field_input = []
        for field in self.l_d_field:
            l_field_input.append(
                Input(shape=(None,),
                      name=pre + self.d_name + '_' + field,
                      dtype='int32')
            )
        return q_input, l_field_input

    def _init_att_input(self, aux=False):
        yield NotImplementedError

    def _init_layers(self):
        self.emb_layer = Embedding(
            self.vocab_size,
            self.embedding_dim,
            weights=[self.emb],
            # mask_zero=True,
            name="embedding",
            trainable=False,
        )
        self.kernel_pool = KernelPooling(np.array(self.mu), np.array(self.sigma), name='kp')
        self.ltr_layer = Dense(
            1,
            name='letor',
            use_bias=False,
            input_dim=len(self.l_d_field) * len(self.mu) + self.ltr_feature_dim
        )
        if self.metric_learning:
            self.distance_metric = Dense(50, input_dim=self.embedding_dim ,use_bias=False)
                # DiagnalMetric(input_dim=self.embedding_dim)

    def _init_ranker(self, q_input, l_field_input, ltr_input=None, aux=False):
        """
        construct ranker for given inputs
        :param q_input:
        :param l_field_input:
        :param ltr_input: if use ltr features to combine
        :param aux:
        :return:
        """
        pre = ""
        if aux:
            pre = self.aux_pre
        q = self.emb_layer(q_input)
        if self.metric_learning:
            q = self.distance_metric(q)
        self.q_emb = q

        l_d_layer = []
        for field, f_in in zip(self.l_d_field, l_field_input):
            d_layer = self.emb_layer(f_in)
            if self.metric_learning:
                d_layer = self.distance_metric(d_layer)
            l_d_layer.append(d_layer)

        # translation matrices
        l_cos_layer = [dot([q, d], axes=-1, normalize=True, name=pre + 'translation_mtx_' + name)
                       for d, name in zip(l_d_layer, self.l_d_field)]

        # kp results of each field
        l_kp_features = [self.kernel_pool(trans_mtx)
                         for trans_mtx, name in zip(l_cos_layer, self.l_d_field)]

        # put features to one vector
        if len(l_kp_features) > 1:
            ranking_features = concatenate(l_kp_features, name=pre + 'ranking_features')
        else:
            ranking_features = l_kp_features[0]

        if ltr_input:
            ranking_features = concatenate([ranking_features, ltr_input],
                                           name=pre + 'ranking_features_with_ltr')

        ranking_layer = self.ltr_layer(ranking_features)
        l_full_inputs = [q_input] + l_field_input
        if ltr_input:
            l_full_inputs.append(ltr_input)
        ranker = Model(inputs=l_full_inputs,
                       outputs=ranking_layer,
                       name=pre + 'ranker')

        if not aux:
            self.l_d_layer = l_d_layer
            self.l_cos_layer = l_cos_layer
            self.l_kp_features = l_kp_features

        return ranker

    def construct_model(self, q_input, l_field_input, l_aux_field_input, ltr_input, aux_ltr_input):
        ranker = self._init_ranker(q_input, l_field_input, ltr_input)
        aux_ranker = self._init_ranker(q_input, l_aux_field_input, aux_ltr_input, True)
        # trainer = concatenate([ranker, aux_ranker])
        # trainer = Lambda(lambda x: x[0] - x[1])(trainer)
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
        assert self.emb is not None
        q_input, l_field_input, l_aux_field_input, ltr_input, aux_ltr_input = self._init_inputs()
        self._init_layers()
        self.ranker, self.trainer = self.construct_model(
            q_input, l_field_input, l_aux_field_input, ltr_input, aux_ltr_input
        )
        return self.ranker, self.trainer


if __name__ == '__main__':
    # unit test

    emb_mtx = np.ones((50, 2))
    for i in xrange(emb_mtx.shape[0]):
        emb_mtx[i, :] = i + 1
    # emb_mtx[0, :] = np.array([-1, 1])
    # emb_mtx[2, 1] = -2
    # emb_mtx[3, 0] = -3
    k_nrm = KNRM()
    k_nrm.set_embedding(emb_mtx)
    # k_nrm.emb[0, 0] = 1
    # k_nrm.mu = [1]
    # k_nrm.sigma = [1]
    q = np.array([[0, 1, 2]])
    k_nrm.l_d_field = ['title']
    title = np.array([[4, 0, 0, 0]])
    aux_title = np.array([[0, 0, 0, 0]])
    h_in = {'q': q, 'd_title': title, 'aux_d_title': aux_title}
    y = np.array([1])
    # k_nrm._init_inputs()
    # ranking_layer = k_nrm._init_layers()

    test_ranker, test_trainer = k_nrm.build()
    print "ranker:"
    test_ranker.summary()
    print "trainer"
    test_trainer.summary()

    print "q embedding"
    model = Model(inputs=k_nrm.q_input, outputs=k_nrm.q_emb)
    model.summary()
    print model.predict(q)
    # #
    print 'd embedding'
    model = Model(inputs=k_nrm.l_field_input[0], outputs=k_nrm.l_d_layer[0])
    print model.predict(title)

    print "translation mtx"
    model = Model(inputs=[k_nrm.q_input] + k_nrm.l_field_input, outputs=k_nrm.l_cos_layer[0])
    trans_mtx = model.predict(h_in)
    print trans_mtx
    print trans_mtx.shape
    #
    print "kp res:"
    model = Model(inputs=[k_nrm.q_input] + k_nrm.l_field_input, outputs=k_nrm.l_kp_features[0])
    kp = model.predict(h_in)
    print kp
    print kp.shape
    #
    # print 'aux kp res:'
    # model = Model(inputs=[k_nrm.q_input] + k_nrm.l_field_input, outputs=k_nrm.l_kp_features[0])
    # kp = model.predict([q, aux_title])
    # print kp
    # print kp.shape

    # trainer.compile('nadam', loss='hinge')
    # trainer.fit(h_in, np.array([-1]))
    test_trainer.compile('nadam', loss='hinge')
    test_trainer.fit(h_in, y, epochs=10, verbose=2)

