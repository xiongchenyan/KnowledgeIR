"""
from char offset to token offset
"""

import json
import sys
from knowledge4ir.utils import (
    QUERY_FIELD,
    TARGET_TEXT_FIELDS
)
l_target_fields = TARGET_TEXT_FIELDS + [QUERY_FIELD]


def make_char_to_token_mapping(text):
    h_map = {}
    return h_map


def convert_offset(h_info):
    for field in h_info:
        text = h_info[field]
        h_char_to_token_loc = make_char_to_token_mapping(text)

