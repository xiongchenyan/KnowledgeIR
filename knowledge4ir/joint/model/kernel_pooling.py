"""
the kernel-pooling layer implementation for keras

"""

import keras.backend as TF
import numpy as np


def kernel_pooling(translation_matrix, l_mean, l_sigma):
    """
    function for Lambda layer
    kernel pooling layer
    :param translation_matrix: input translation matrix
    :param l_mean: kp's mean
    :param l_sigma: sigma for kernels
    :return: output_shape = 1 + (hyper_parameter.l_kernel_pool_mean) # added 1 exact match pool
    """
    #TODO
    assert len(l_mean) == len(l_sigma)

    mu = np.array(l_mean)  # add exact match kernel
    mu.reshape((1, 1, len(l_mean)))
    sigma = np.array(l_sigma)
    sigma.reshape((1, 1, len(l_sigma)))

    m = TF.expand_dims(translation_matrix, -1)

    raw_k_pool = TF.exp(
        TF.div(
            TF.negative(TF.square(TF.sub(m, mu))),
            TF.mul(TF.square(sigma), 2)
        )
                   )
    k_pool = TF.reduce_sum(raw_k_pool, [0, 1])
    k_pool = TF.log(TF.maximum(k_pool, 1e-10)) * 0.01

    return k_pool

