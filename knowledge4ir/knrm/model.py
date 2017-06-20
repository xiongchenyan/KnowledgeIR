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
    aux_pre = Unicode('aux')
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
        q_input, l_field_input = self._init_rank_input()
        __, l_aux_field_input = self._init_rank_input(aux=True)
        self.q_input = q_input
        self.l_field_input = l_field_input
        self.l_aux_field_input = l_aux_field_input
        return q_input, l_field_input, l_aux_field_input

    def _init_rank_input(self, aux=False):
        pre = ""
        if aux:
            pre = self.aux_pre
        q_input = Input(shape=(self.q_len,), name=pre + self.q_name, dtype='int32')
        l_field_input = []
        for field, f_len in zip(self.l_d_field, self.l_d_field_len):
            l_field_input.append(
                Input(shape=(f_len,),
                      name=pre + self.d_name + '_' + field,
                      dtype='int32')
            )
        return q_input, l_field_input

    def _init_att_input(self, aux=False):
        yield NotImplementedError

    def _init_layers(self):
        """
        layers:
            embedding
            distance metric (TODO)
            l2 normalize
            dot merge (to translation matrix)
            kernel pooling
            Dense
        :return:
        """
        self.emb_layer = Embedding(
            len(self.vocab_size),
            self.embedding_dim,
            weights=[self.emb],
            # mask_zero=True,
            name="embedding_layer"
        )
        q = self.emb_layer(Input(shape=(None,), dtype='int32'))

        l_d_layer = []
        for field, f_len in zip(self.l_d_field, self.l_d_field_len):
            d_layer = self.emb_layer(Input(shape=(None, ), dtype='int32'))
            l_d_layer.append(d_layer)

        # translation matrices
        self.l_cos_layer = [dot(q, d, axes=-1, normalize=True, name='translation_mtx_%s' % name) for d, name in zip(l_d_layer, self.l_d_field)]

        # kp results of each field
        l_kp_features = [KernelPooling(np.array(self.mu), np.array(self.sigma), name='kp_%s' % name)(trans_mtx)
                         for trans_mtx, name in zip(self.l_cos_layer, self.l_d_field)]

        # put features to one vector
        ranking_features = concatenate(l_kp_features, name='ranking_features')
        ranking_layer = Dense(1, name='letor')(ranking_features)
        self.ranking_layer = ranking_layer
        return ranking_layer

    def _pack_layer_to_ranker(self, q_input, l_field_input, ranking_layer):
        ranker = Model(input=[q_input] + l_field_input, output=ranking_layer)
        return ranker

    def _pack_layer_to_trainer(self, q_input, l_field_input, l_aux_field_input, ranking_layer):

        ranker = self._pack_layer_to_ranker(q_input, l_field_input, ranking_layer)
        aux_ranker = self._pack_layer_to_ranker(q_input, l_aux_field_input,
                                                ranking_layer)
        trainer = Sequential()
        trainer.add(concatenate([ranker, aux_ranker]))
        trainer.add(Lambda(lambda x: x[0] - x[1]))
        return ranker, trainer

    def build(self):
        assert self.emb
        q_input, l_field_input, l_aux_field_input = self._init_inputs()
        ranking_layer = self._init_layers()
        self.ranker, self.trainer = self._pack_layer_to_trainer(
            q_input, l_field_input, l_aux_field_input, ranking_layer
        )
        return self.ranker, self.trainer


if __name__ == '__main__':
    # unit test

    emb_mtx = np.ones((50, 16))
    for i in xrange(emb_mtx.shape[0]):
        emb_mtx[i, :] = i
    k_nrm = KNRM()
    k_nrm.set_embedding(emb_mtx)
    q = np.array([0, 1])
    k_nrm.l_d_field = ['title']
    title = np.array([2, 3, 4])
    h_in = {'q': q, 'd_title': title}

    k_nrm._init_inputs()
    ranking_layer = k_nrm._init_layers()

    print "q embedding"
    model = Sequential(k_nrm.emb_layer)
    model.summary()
    print model.predict(q)

    print 'd embedding'
    print model.predict(title)

    print "translation mtx"
    model = Model(input=[k_nrm.q_input] + k_nrm.l_field_input, output=k_nrm.l_cos_layer[0])
    print model.predict(h_in)
    ranker, trainer = k_nrm.build()
    print "ranker:"
    ranker.summary()
    print "trainer"
    trainer.summary()






