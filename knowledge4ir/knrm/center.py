"""
KNRM center
Implement the API's defined in model.base.ModelBase
keras model constructed via knrm.model
data i/o implemented in knrm.data_reader
hyper-parameter maintained via model.hyper_parameter
"""


from knowledge4ir.model.base import ModelBase
from knowledge4ir.model.hyper_para import HyperParameter
from knowledge4ir.knrm.model import KNRM
from knowledge4ir.knrm.data_reader import (
    pairwise_reader,
    pointwise_reader
)
import json
import logging
from traitlets import (
    Unicode,
    Int,
    List,
)
import numpy as np


class KNRMCenter(ModelBase):
    model_name = Unicode('KRNM')
    qrel_in = Unicode().tag(config=True)
    q_info_in = Unicode().tag(config=True)
    doc_info_in = Unicode().tag(config=True)
    embedding_npy_in = Unicode().tag(config=True)
    
    def __init__(self, **kwargs):
        super(KNRMCenter, self).__init__(**kwargs)
        self.k_nrm = KNRM(**kwargs)
        emb_mtx = np.load(self.embedding_npy_in)
        self.k_nrm.set_embedding(emb_mtx)
        self.ranker, self.learner = self.k_nrm.build()


    def train(self, x, y, hyper_para=None):
        
        yield NotImplementedError

    def predict(self, x):
        yield NotImplementedError

    def train_data_reader(self, in_name, s_target_qid=None):
        yield NotImplementedError

    def test_data_reader(self, in_name, s_target_qid=None):
        yield NotImplementedError


