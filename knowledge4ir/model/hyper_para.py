"""
hyper_parameter class
"""
import json

from traitlets import Float, Tuple, Int, List, Unicode
from traitlets.config import Configurable


class HyperParameter(Configurable):
    # model parameters
    l2_w = Float(0.01).tag(config=True)
    dropout_rate = Float(0).tag(config=True)
    q_shape = Tuple(Int, default_value=(5, 1)).tag(config=True)
    title_shape = Tuple(Int, default_value=(10, 1)).tag(config=True)
    body_shape = Tuple(Int, default_value=(300, 1)).tag(config=True)
    embedding_dim = Int(300).tag(config=True)
    l_kernel_pool_mean =List(Float, default_value=[],
                             help='kernal pooling means'
                             ).tag(config=True)
    l_kernel_pool_sigma = List(Float, default_value=[],
                               help='kernal pooling sigmas'
                               ).tag(config=True)

    # training parameters
    loss = Unicode('hinge').tag(config=True)
    opt = Unicode('nadam').tag(config=True)
    batch_size = Int(-1).tag(config=True)
    nb_epoch = Int(10).tag(config=True)
    early_stopping_patient = Int(10).tag(config=True)

    def pretty_print(self):
        """
        only print those to be explored
        :return:
        """
        h_target = {}
        l_target = ['l2_w', 'dropout_rate']
        for target in l_target:
            value = getattr(self, target)
            h_target[target] = value
        return json.dumps(h_target)


if __name__ == '__main__':
    HyperParameter.class_print_help()
