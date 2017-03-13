"""
base matching feature extraction class

defines the API of matching classes

"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
    Int
)
import json
import logging


class MatchFeatureExtractor(Configurable):
    feature_name_pre = Unicode(help='set the feature name pre here')

    def extract(self, q_info, d_info, external_resource):
        raise NotImplementedError



