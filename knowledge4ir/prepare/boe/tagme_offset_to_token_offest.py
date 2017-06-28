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
    term_p = 0
    for i in xrange(len(text) + 1):
        if i == ' ':
            term_p += 1
        h_map[i] = term_p
    return h_map


def convert_offset(h_info):
    for field in l_target_fields:
        if field not in h_info:
            continue
        text = h_info[field]
        h_char_to_token_loc = make_char_to_token_mapping(text)
        l_ana = h_info['tagme'][field]
        for i in xrange(len(l_ana)):
            loc = l_ana[i][1:3]
            l_ana[i][1], l_ana[i][2] = h_char_to_token_loc[loc[0]], h_char_to_token_loc[loc[1]]
    return h_info

if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "convert offset from char to token in tagme's ana"
        print "2 para: tagge info in + out"
        sys.exit(-1)
    out = open(sys.argv[2], 'w')
    for p, line in enumerate(open(sys.argv[1])):
        if not p % 100:
            print "converted [%d] lines" % p
        h = json.loads(line)
        h = convert_offset(h)
        print >> out, json.dumps(h)
    out.close()
    print "done"
