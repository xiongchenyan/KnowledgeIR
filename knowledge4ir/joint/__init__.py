"""
overall architecture:
1) spotter: add top 3 candidate entities for each position
    a) entity linking' spotter
    b) coreference spotter for pronouns
2) grounding
    a) extracts features for each candidate entities
        entity linking feature
        coreference feature
        query's entity linking feature also comes from the PRF docs
3) model preparation
    a) convert the features into keras' input format
    b) map the entities ids into interger, prepare embeddings matrix

4) Ranking model


"""

from .base import *
