"""
soft ESR model
takes the soft-entities sequence of q and d as input
uses embedding as the external resource
uses hyperparameter class input
"""

from keras.layers import (
    Merge,
    Lambda,
    Dense,
    Input,
    Embedding,
)
from keras.regularizers import (
    l2,
)
from keras.models import (
    Model
)
from knowledge4ir.joint import (
    JointSemanticModel,
    JointSemanticResource,
)

from traitlets.config import Configurable
from traitlets import (
    Float,
    Int,
    Tuple,
    Unicode
)


class SoftESR(JointSemanticModel):


    def _build_para_layers(self):
        #TODO
        return

    def _form_model_from_layers(self, h_para_layers):
        #TODO
        return