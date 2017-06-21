"""
data i/o
input:
    trec rank candidate docs
    qrels
    q info (tensor format)
    doc info (tensor format)
output:
    pairwise x, y (for target q id)
    pointwise x, y (for target q id)
"""

from knowledge4ir.knrm import (
    aux_pre,
    q_in_name,
    d_in_name,
    ltr_feature_name,
    q_att_name,
    d_att_name
)
from knowledge4ir.utils import (
    load_trec_labels_dict,
    load_trec_ranking_with_score,
    load_json_info,
)

import json
import logging


