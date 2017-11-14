"""
unit test
TODO
"""

import logging

import numpy as np
from traitlets import Unicode
from traitlets.config import Configurable

from knowledge4ir.salience.baseline.translation_model import GraphTranslation


class ModuleTester(Configurable):

    h_model = {
        "trans": GraphTranslation
    }
    embedding_in = Unicode(help='entity embedding').tag(config=True)
    model_name = Unicode("trans", help='model name').tag(config=True)

    def __init__(self, **kwargs):
        super(ModuleTester, self).__init__(**kwargs)
        self.emb_mtx = np.zeros(0)
        if self.embedding_in:
            logging.info('loading embedding mtx [%s]', self.embedding_in)
            self.emb_mtx = np.load(open(self.embedding_in))
        self.model = self.h_model[self.model_name](
            1, self.emb_mtx.shape[0], self.emb_mtx.shape[1], self.emb_mtx
        )
        logging.info('model [%s] initialized', self.model_name)

    def run_forward(self):
        """
        check one forward run
        :return:
        """



