"""
analysis ranking component of qw_de qe_de
input:
    q info
    trec rank with doc info
output:
    for each q-d pair:
        top k entity search score's entity (Lm dir on description)
        for each query entity, doc's entity in each transE bin
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
)
from knowledge4ir.feature import LeToRFeatureExternalInfo
from knowledge4ir.utils import (
    load_query_info,
    load_trec_ranking_with_info,
)
import numpy as np
import json
import logging


class RankComponentAna(Configurable):
    
    def __init__(self, **kwargs):
        super(RankComponentAna, self).__init__(**kwargs)

