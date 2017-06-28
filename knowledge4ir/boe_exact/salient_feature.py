"""
simple salient features, except frequency

version 1 (06282017)
position,
uw vote from w and from e
    pooled to one score
"""

from knowledge4ir.utils.boe import (
    uw_word_embedding_vote,
    entity_embedding_vote,
    word_embedding_vote,
)
import json
import logging
from knowledge4ir.boe_exact.boe_feature import BoeFeature



