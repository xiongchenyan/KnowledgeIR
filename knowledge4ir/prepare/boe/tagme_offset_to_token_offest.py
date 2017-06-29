"""
from char offset to token offset
"""

import json
import sys
from knowledge4ir.utils import (
    QUERY_FIELD,
    TARGET_TEXT_FIELDS
)
import logging
l_target_fields = TARGET_TEXT_FIELDS + [QUERY_FIELD]


def make_char_to_token_mapping(text):
    h_map = {}
    term_p = 0
    for i in xrange(len(text)):
        if text[i] == ' ':
            term_p += 1
        h_map[i] = term_p
    h_map[len(text)] = term_p + 1
    return h_map


def convert_offset(h_info):
    for field in l_target_fields:
        if field not in h_info:
            continue
        text = h_info[field]
        h_char_to_token_loc = make_char_to_token_mapping(text)
        if 'tagme' in h_info:
            l_ana = h_info['tagme'].get(field, [])
        else:
            l_ana = h_info['spot'].get(field, [])
        for i in xrange(len(l_ana)):
            if 'loc' in l_ana[i]:
                loc = l_ana[i]['loc']
                st, ed = h_char_to_token_loc[loc[0]], h_char_to_token_loc[loc[1] - 1]
                ed += 1
                # if text[ed] != " ":
                    # the non-English char ed offset?
                    # ed += 1
                l_ana[i]['loc'] = (st, ed)
            else:
                loc = l_ana[i][1:3]
                st, ed = h_char_to_token_loc[loc[0]], h_char_to_token_loc[loc[1] - 1]
                ed += 1
                l_ana[i][1], l_ana[i][2] = st, ed
            before_name = text[loc[0]: loc[1]]
            after_name = ' '.join(text.split()[st:ed])
            if after_name not in before_name:
                logging.warn('location match: [%s] -> [%s]', before_name, after_name)

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
