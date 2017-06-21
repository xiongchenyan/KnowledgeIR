"""
Kernel Pooling layer in keras
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
        self.mu = np.array(mu, dtype='float32')
        self.sigma = np.array(sigma, dtype='float32')
        # assert mu.shape == sigma.shape
        self.nb_k = len(mu)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.nb_k

    def call(self, inputs, **kwargs):
        """
        for each of input's last dimension (x)
        exp ((x - mu) * 2 / (2 * sigma*2))
        :param inputs:
        :return:
        """

        # broad cast, d0: batch, d1: q, d2: doc, d3: kernel

        m = K.expand_dims(inputs, -1)

        sq_diff = -K.square(m - self.mu)
        mod = 2.0 * K.square(self.sigma)
        raw_k_pool = K.exp(sq_diff / mod)

        # sum up the document dimension
        # from batch, q, doc, kernel to batch, q, kernel

        k_pool = K.sum(raw_k_pool, 2)

        # log sum along the q axis
        # from batch, q, k to batch, k
        k_pool = K.sum(K.log(k_pool), 1)
        # k_pool = shared(self.mu.reshape(1, self.mu.shape[0]))
        return k_pool


if __name__ == '__main__':
    x = np.array([[[1, 1, 1], [0, 1, 2]], [[2, 2, 2], [0, 2, 4]]])
    from keras.models import Sequential

    m = Sequential()
    mu = np.array([1,2])
    sigma = np.array([2, 2])
    m.add(
        KernelPooling(
            mu, sigma,
            input_shape=(2, None)
        )
    )
    y = m.predict(x)
    print y.tolist()
