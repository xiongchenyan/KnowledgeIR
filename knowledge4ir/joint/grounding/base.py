"""
the base class for grounding features
input:
    a json dict of q or doc
        with information packed
            "spot"
            "query"
            "fields"
    the external resource
        surface form dict (multiple?)
        surface form link p
        e embedding
        for PRF: q's d's info
do:
    extract grounding features for surface form: link or not
    extract grounding features for entities: whether it is to link to the current one
"""

import json
import logging
from traitlets.config import Configurable
from traitlets import (
    List,
    Unicode,
    Int,
)
from knowledge4ir.joint.resource import JointSemanticResource


class Grounder(Configurable):
    feature_pre = Unicode()

    def __init__(self, **kwargs):
        super(Grounder, self).__init__(**kwargs)
        self.resource = JointSemanticResource(**kwargs)

    def set_resource(self, external_resource):
        self.resource = external_resource

    def extract(self, h_info):
        """
        extract and add features for surfaces and entities in h_info['spot']
        :param h_info:
        :return: packed into h_info
        """
        return h_info

    def extract_for_surface(self, h_sf_info, h_info):
        """

        :param h_sf_info: {sf, st, ed, field}
        :param h_info: same as extract's h_info
        :return: h_feature, features for this sf
        """
        raise NotImplementedError

    def extract_for_entity(self, h_e_info, h_sf_info, h_info):
        """

        :param h_e_info: e info
        :param h_sf_info: sf info
        :param h_info: total data info
        :return: h_feature for this entity
        """
        raise NotImplementedError





