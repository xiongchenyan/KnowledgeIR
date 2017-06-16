"""
q info and doc info basic operations
"""

import json
from knowledge4ir.joint import SPOT_FIELD
from knowledge4ir.utils import (
    QUERY_FIELD,
    TARGET_TEXT_FIELDS,
)


def form_boe_per_field(h_info, field):
    l_ana = h_info.get(SPOT_FIELD, {}).get(field, [])
    l_e = []
    for ana in l_ana:
        sf = ana['surface']
        loc = ana['loc']
        e = ana['entities'][0]
        h = {'surface': sf, 'loc': loc, 'id': e}
        l_e.append(h)
    return l_e


def form_boe_tagme_field(h_info, field):
    l_ana = h_info.get('tagme', {}).get(field, [])
    l_e = []
    for ana in l_ana:
        e, st, ed = ana[:3]
        loc = (st, ed)
        sf = ana[-1]
        h = {'surface': sf, 'loc': loc, 'id': e}
        l_e.append(h)
    return l_e
