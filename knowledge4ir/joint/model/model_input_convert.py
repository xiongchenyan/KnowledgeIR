"""
convert json style model input (grounding and matchign features)
to numpy ready format
    json, but aligned and qrels feature matrices (list in the disk)

input:
    grounded q json
    matched q-d pairs (match fields ready)
        make sure the features are all full (no missing feature values)
    qrel
    max sf per q (3)
    max e per sf (3)
output:
    one line per pair
        label:
        sf matrix: |spot|*|f dim|
        sf-entity ground: (tensor) |spot||candidate entity||f dim|
        sf_entity qe_d: |spot||candidate entity||f dim|
        all the three's corresponding dimension are aligned
        meta:
            sf: list of loc for sf matrix first dim
            sf-entity matrix: list of sf for sf_entity
            qid
            docno
"""


import json
import sys
from knowledge4ir.utils import (
    load_trec_labels_dict
)
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)


class ModelInputConvert(Configurable):
    
    
    def __init__(self, **kwargs):
        super(ModelInputConvert, self).__init__(**kwargs)






