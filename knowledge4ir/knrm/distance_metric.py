"""
keras layer that aims to learn a distance metric w on the embedding
"""

from keras import backend as K
from keras.engine.topology import Layer, InputSpec, initializers
import numpy as np


class DiagnalMetric(Layer):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DiagnalMetric, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.metric = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.metric = self.add_weight(shape=(input_dim, ),
                                      initializer=self.kernel_initializer,
                                      name='metric',
                                     )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        output = inputs * (K.clip(self.metric, -1, 1) + 1.1)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return input_shape
