"""
construct the entity grid
sentence -> e id

input:
    spotted documents
    with texts, and annotations
output:
    add the entity_grid field to the dict:
    entity_grid -> fields
        -> list of sentences -> ana with location altered to the sentence's offset

    can directly dump pretty print format (with each line one sentence)
        docno \t sentence: list of ana in each line
"""

from knowledge4ir.utils.boe import form_boe_per_field
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    QUERY_FIELD,
)
import json
from nltk.tokenize import sent_tokenize


def construct_per_text(text, l_ana):
    """

    :param text:
    :param l_ana: list of ana in form_boe_per_field's format
    :return: l_e_grid = [ {'sent':, 'spot':}  ]
    """

    l_sent = sent_tokenize(text)
    ll_ana = _align_ana_to_sent(l_sent, l_ana)
    l_e_grid = []
    for sent, l_ana in zip(l_sent, ll_ana):
        pass

    return l_e_grid

def _align_ana_to_sent(l_sent, l_ana):
    pass