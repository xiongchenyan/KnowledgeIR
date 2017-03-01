"""
model's hyper parameters
base functions
"""

from traitlets.config import Configurable
from traitlets import (
    Float,
    Int,
    Tuple,
    Unicode,
    List,
)
from knowledge4ir.joint.resource import JointSemanticResource


def kernel_pooling(translation_matrix, hyper_parameter):
    """
    function for Lambda layer
    kernel pooling layer
    :param translation_matrix: input translation matrix
    :param hyper_parameter: HyperParameter()
    :return:
    """
    #TODO
    return


class HyperParameter(Configurable):
    l2_w = Float(0.01).tag(config=True)
    dropout_rate = Float(0).tag(config=True)
    q_shape = Tuple(Int, default_value=(5, 1)).tag(config=True)
    title_shape = Tuple(Int, default_value=(10, 1)).tag(config=True)
    body_shape = Tuple(Int, default_value=(300, 1)).tag(config=True)
    embedding_dim = Int(300).tag(config=True)
    l_kernel_pool_mean =List(Float, default_value=[],
                             help='will always add the exact kernel'
                             ).tag(config=True)
    kernel_pool_lambda = Float(0.1).tag(config=True)


class JointSemanticModel(Configurable):
    """
    the base class for all models
    """
    aux_pre = Unicode('aux_')
    q_name = Unicode('q')
    q_att = Unicode('q_att')
    title_name = Unicode('title')
    body_name = Unicode('body')
    title_att = Unicode('title_att')
    body_att = Unicode('body_att')

    def __init__(self, **kwargs):
        super(JointSemanticModel, self).__init__(**kwargs)
        self.hyper_para = HyperParameter(**kwargs)
        self.external_resource = JointSemanticResource(**kwargs)

        self.ranking_model = None
        self.training_model = None

    def set_external_resource(self, resource):
        self.external_resource = resource

    def _build_model(self):
        h_para_layers = self._build_para_layers()
        self.ranking_model, self.training_model = self._form_model_from_layers(h_para_layers)
        return

    def _build_para_layers(self):
        raise NotImplementedError

    def _form_model_from_layers(self, h_para_layers):
        raise NotImplementedError

    def pairwise_train(self, l_paired_data, Y):
        """
        pairwise training
        :param l_paired_data: each element is a pair of doc's, Y is their order (1 or -1, for the order)
        :param Y: label
        :return: trained model
        """
        #TODO
        return

    def predict(self, l_data):
        #TODO
        return

