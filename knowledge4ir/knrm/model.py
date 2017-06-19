"""
the model part of knrm

KP layer

and KNRM model ranking inits
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class KernelPooling(Layer):
    """
    parameters:
    mu: the array of mu's, d * 1
    sigma: the array of sigma's, d * 1
    input shape: a translation matrix: |q| * |d|
    output shape is the batch * input_shape's first dimension |q| * size of mu
    """
    
    def __init__(self, mu, sigma, **kwargs):
        super(KernelPooling, self).__init__(**kwargs)
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def call(self, inputs, **kwargs):
        """
        for each of input's last dimension (x)
        exp ((x - mu) * 2 / (2 * sigma*2))
        :param inputs:
        :return:
        """
        output = None
        m = K.expand_dims(inputs, -1)

        sq_diff = -K.square(m - self.mu)
        mod = 2 * K.square(self.sigma)
        raw_k_pool = K.exp(sq_diff / mod)

        # sum up the document dimension
        k_pool = K.sum(raw_k_pool, 3)

        # log sum along the q axis
        k_pool = K.sum(K.log(k_pool), 1)

        return k_pool




