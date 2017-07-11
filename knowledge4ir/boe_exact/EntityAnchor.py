"""
Entity Anchored Semi-structure doc representation
    to get better LeToR features

In the pipeline, form as a subclass of NLSSFeature

do:
    construct entity Anchored Semi-structure representation
    sentence -> entity, only if query entity is in the sentence
        the scores include:
            embedding sim
            bow sim (frequency)
            embedding sim vs desp
            bow sim vs desp

    extract features from the constructed grid:
        pooling grid scores to features
        entity proximity
        word proximity (another class)
        full letor (another class)
"""

from traitlets import (
    Unicode,
    Int,
    List,
)
from knowledge4ir.boe_exact.boe_feature import BoeFeature
from knowledge4ir.utils import (
    QUERY_FIELD,
    TARGET_TEXT_FIELDS,
    lm_cosine,
    avg_embedding,
    body_field,
)
from knowledge4ir.utils.retrieval_model import RetrievalModel


class EntityAnchorFeature(BoeFeature):
    feature_name_pre = Unicode('EA')
    l_target_fields = List(Unicode, default_value=[body_field]).tag(config=True)
    gloss_len = Int(15, help='gloss length').tag(config=True)
    max_grid_sent_len = Int(100, help='max grid sentence len to consider').tag(config=True)

    def set_resource(self, resource):
        self.resource = resource
        assert self.resource.l_h_desp
        assert self.resource.embedding

    def extract_per_entity(self, q_info, ana, doc_info):
        """
        extract per entity feature
        :param q_info:
        :param ana:
        :param doc_info:
        :return:
        """
        h_feature = {}
        return h_feature

