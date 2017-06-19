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
    mu = List(Float, help='mu of kernel pooling').tag(config=True)
    sigma = List(Float, help='sigma of kernel pooling').tag(config=True)

    def __init__(self, **kwargs):
        super(KNRM, self).__init__(**kwargs)
        self.emb = None

        self.q_input = None
        self.l_field_input = []
        self.l_aux_field_input = []

        self.ranking_layer = None
        self.ranker = None
        self.trainer = None
        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

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
                      name=pre + field + '_' + self.d_name,
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
        emb_layer = Embedding(
            len(self.vocab_size + 1),
            self.embedding_dim,
            weights=[self.emb],
            mask_zero=True,
        )
        q = emb_layer(Input(shape=(None,), dtype='int32'))

        l_d_layer = []
        for field, f_len in zip(self.l_d_field, self.l_d_field_len):
            d_layer = emb_layer(Input(shape=(None, ), dtype='int32'))
            l_d_layer.append(d_layer)

        # translation matrices
        l_cos_layer = [dot(q, d, axes=-1, normalize=True) for d in l_d_layer]

        # kp results of each field
        l_kp_features = [KernelPooling(self.mu, self.sigma)(trans_mtx)
                         for trans_mtx in l_cos_layer]

        # put features to one vector
        ranking_features = concatenate(l_kp_features)
        ranking_layer = Dense(1)(ranking_features)
        self.ranking_layer = ranking_layer
        return ranking_layer

    def _pack_layer_to_ranker(self, q_input, l_field_input, ranking_layer):
        ranker = Model(inputs=[q_input] + l_field_input, outputs=ranking_layer)
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







