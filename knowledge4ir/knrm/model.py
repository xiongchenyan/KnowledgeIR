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
)
from keras.models import (
    Sequential,
    Model
)
from knowledge4ir.knrm.kernel_pooling import KernelPooling
import numpy as np
from traitlets.config import Configurable
from traitlets import (
    Bool,
    Int,
    Unicode,
    List,
    Float,
)
import keras.backend as K
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
)
from keras.layers.normalization import BatchNormalization


class KNRM(Configurable):
    """
    init knrm model
    ranking and training together
    """
    embedding_dim = Int(50, help='embedding dim').tag(config=True)
    vocab_size = Int(help='vocab size').tag(config=True)
    q_name = Unicode('q')
    d_name = Unicode('d')
    d_att_name = Unicode('d_att')
    q_att_name = Unicode('q_att')
    q_len = Int(5, help='maximum q entity length')
    aux_pre = Unicode('aux_')
    l_d_field = List(Unicode, default_value=TARGET_TEXT_FIELDS,
                     help='fields in the documents').tag(config=True)
    l_d_field_len = List(Int, default_value=[10, 500],
                         help='max len of each field').tag(config=True)
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
        # self.mu = np.array(self.mu)
        # self.sigma = np.array(self.sigma)

    def set_embedding(self, pretrained_emb):
        self.emb = pretrained_emb
        self.vocab_size, self.embedding_dim = pretrained_emb.shape

    def _init_inputs(self):
        q_input, l_field_input = self._init_one_side_input()
        __, l_aux_field_input = self._init_one_side_input(aux=True)
        self.q_input = q_input
        self.l_field_input = l_field_input
        self.l_aux_field_input = l_aux_field_input
        return q_input, l_field_input, l_aux_field_input

    def _init_one_side_input(self, aux=False):
        pre = ""
        if aux:
            pre = self.aux_pre
        q_input = Input(shape=(None,), name=pre + self.q_name, dtype='int32')
        l_field_input = []
        for field, f_len in zip(self.l_d_field, self.l_d_field_len):
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
        self.ltr_layer = Dense(1, name='letor', use_bias=False, input_dim=len(self.l_d_field) * len(self.mu))

    def _init_ranker(self, q_input, l_field_input, aux=False):
        pre = ""
        if aux:
            pre = self.aux_pre
        q = self.emb_layer(q_input)
        self.q_emb = q
        l_d_layer = []
        for field, f_in in zip(self.l_d_field, l_field_input):
            d_layer = self.emb_layer(f_in)
            l_d_layer.append(d_layer)

        # translation matrices
        l_cos_layer = [dot([q, d], axes=-1, normalize=True, name=pre + 'translation_mtx_' + name)
                            for d, name in zip(l_d_layer, self.l_d_field)]

        # kp results of each field
        l_kp_features = [self.kernel_pool(trans_mtx)
                         for trans_mtx, name in zip(l_cos_layer, self.l_d_field)]

        # put features to one vector
        if len(l_kp_features) > 1:
            ranking_features = concatenate(l_kp_features, name= pre + 'ranking_features')
        else:
            ranking_features = l_kp_features[0]
        ranking_layer = self.ltr_layer(ranking_features)
        ranker = Model(inputs=[q_input] + l_field_input, outputs=ranking_layer, name=pre + 'ranker')

        if not aux:
            self.l_d_layer = l_d_layer
            self.l_cos_layer = l_cos_layer
            self.l_kp_features = l_kp_features

        return ranker

    def construct_model(self, q_input, l_field_input, l_aux_field_input):

        ranker = self._init_ranker(q_input, l_field_input)
        aux_ranker = self._init_ranker(q_input, l_aux_field_input, True)
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
        q_input, l_field_input, l_aux_field_input = self._init_inputs()
        self._init_layers()
        self.ranker, self.trainer = self.construct_model(
            q_input, l_field_input, l_aux_field_input
        )
        return self.ranker, self.trainer


if __name__ == '__main__':
    # unit test

    emb_mtx = np.ones((50, 2))
    for i in xrange(emb_mtx.shape[0]):
        emb_mtx[i, :] = i
    emb_mtx[0, :] = np.array([-1, 1])
    emb_mtx[2,1] = -2
    emb_mtx[3,0] = -3
    k_nrm = KNRM()
    k_nrm.set_embedding(emb_mtx)
    # k_nrm.mu = [1]
    # k_nrm.sigma = [1]
    q = np.array([[0, 1]])
    k_nrm.l_d_field = ['title']
    title = np.array([[2, 3, 4]])
    aux_title = np.array([[0, 1]])
    h_in = {'q': q, 'd_title': title, 'aux_d_title': aux_title}

    # k_nrm._init_inputs()
    # ranking_layer = k_nrm._init_layers()

    ranker, trainer = k_nrm.build()
    print "ranker:"
    ranker.summary()
    print "trainer"
    trainer.summary()
    #
    # print "q embedding"
    # model = Model(inputs=k_nrm.q_input, outputs=k_nrm.q_emb)
    # model.summary()
    # print model.predict(q)
    # #
    # print 'd embedding'
    # model = Model(inputs=k_nrm.l_field_input[0], outputs=k_nrm.l_d_layer[0])
    # print model.predict(title)
    #
    # print "translation mtx"
    # model = Model(inputs=[k_nrm.q_input] + k_nrm.l_field_input, outputs=k_nrm.l_cos_layer[0])
    # trans_mtx = model.predict(h_in)
    # print trans_mtx
    # print trans_mtx.shape
    #
    # print "kp res:"
    # model = Model(inputs=[k_nrm.q_input] + k_nrm.l_field_input, outputs=k_nrm.l_kp_features[0])
    # kp = model.predict(h_in)
    # print kp
    # print kp.shape
    #
    # print 'aux kp res:'
    # model = Model(inputs=[k_nrm.q_input] + k_nrm.l_field_input, outputs=k_nrm.l_kp_features[0])
    # kp = model.predict([q, aux_title])
    # print kp
    # print kp.shape

    # trainer.compile('nadam', loss='hinge')
    # trainer.fit(h_in, np.array([-1]))
    ranker.compile('nadam', loss='hinge')
    ranker.fit(h_in, np.array([1]))

