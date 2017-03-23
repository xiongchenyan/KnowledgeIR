"""
provide pipe line api to run various models
the model must be defined in the h_name_model
the model must be follow the API defined by ModelBase()

the hyper_parameters class is obtained via model_pipe.hyper_para

run one fold: train-test
run with given train_in and test_in, can be used to
run one fold with dev: train-dev-test

fold split by qid % k

"""

from traitlets import (
    Unicode,
)
from traitlets.config import (
    Configurable,
)

class ModelBase(Configurable):
    model_name = Unicode('base_model')

    def train(self, x, y, hyper_para=None):
        yield NotImplementedError

    def predict(self, x):
        yield NotImplementedError

    def train_with_dev(self, x, y, dev_x, dev_y, l_hyper_para):
        yield NotImplementedError

    def generate_ranking(self, x, out_name):
        yield NotImplementedError

    def train_data_reader(self, in_name, s_target_qid=None):
        yield NotImplementedError

    def test_data_reader(self, in_name, s_target_qid=None):
        yield NotImplementedError


