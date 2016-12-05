"""
attention feature base
base class with API defined, and data set
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List
)
import logging


class TermAttentionFeature(Configurable):
    feature_name_pre = Unicode('TAtt')

    def set_external_info(self, external_info):
        logging.info('setting external info with shared storage')
        return

    def extract(self, h_q_info, l_t):
        yield NotImplementedError


class EntityAttentionFeature(Configurable):
    feature_name_pre = Unicode('EAtt')

    def set_external_info(self, external_info):
        logging.info('setting external info with shared storeage')
        return

    def extract(self, h_q_info, l_e):
        yield NotImplementedError

