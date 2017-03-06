"""
base
"""

from traitlets.config import Configurable
from traitlets import Unicode
import logging


class QAttFeatureExtractor(Configurable):
    feature_name_pre = Unicode()

    def extract(self, qid, h_info):
        logging.error('extract not implemented')
        yield NotImplementedError
