"""
resource to keep in memory to be shared across the pipeline
"""

import numpy as np
from traitlets.config import Configurable
from traitlets import (
    Unicode,
)
import logging
import json


class JointSemanticResource(Configurable):
    surface_form_path = Unicode(help="the location of surface form dict, in Json format"
                                ).tag(config=True)
    embedding_path = Unicode(help="embedding numpy matrix location"
                             ).tag(config=True)
    
    def __init__(self, **kwargs):
        super(JointSemanticResource, self).__init__(**kwargs)
        self.embedding = np.array(None)
        self.h_surface_form = dict()
        self._load()

    def _load(self):
        self._load_sf()
        self._load_emb()

        return

    def _load_sf(self):
        if not self.surface_form_path:
            return
        logging.info('loading sf dict from [%s]', self.surface_form_path)
        self.h_surface_form = json.load(self.surface_form_path)
        logging.info('sf dict loaded')

    def _load_emb(self):
        if not self.embedding_path:
            return
        logging.info('loading embedding np mtx [%s]', self.embedding_path)
        self.embedding = np.load(self.embedding_path)
        logging.info('embedding array loaded')


