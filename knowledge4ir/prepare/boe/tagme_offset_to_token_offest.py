"""
from char offset to token offset
"""

import json
import sys
from knowledge4ir.utils import (
    QUERY_FIELD,
    TARGET_TEXT_FIELDS,
    abstract_field,
    SPOT_FIELD,
)
import logging


def make_char_to_token_mapping(text):
    h_map = {}
    term_p = 0
    for i in xrange(len(text)):
        if text[i] == ' ':
            term_p += 1
        h_map[i] = term_p
    h_map[len(text)] = term_p + 1
    return h_map


def convert_offset(h_info, conf_para=None):
    l_target_fields = TARGET_TEXT_FIELDS + [QUERY_FIELD, abstract_field]
    if conf_para:
        l_target_fields = para.l_target_fields

    for field in l_target_fields:
        if field not in h_info:
            continue
        text = h_info[field]
        h_char_to_token_loc = make_char_to_token_mapping(text)
        if not conf_para:
            if 'tagme' in h_info:
                l_ana = h_info['tagme'].get(field, [])
            else:
                l_ana = h_info['spot'].get(field, [])
        else:
            l_ana = h_info[para.spot_field].get(field, [])
        min_st = 0
        l_new_ana = []
        for i in xrange(len(l_ana)):
            if 'loc' in l_ana[i]:
                loc = l_ana[i]['loc']
            else:
                loc = l_ana[i][1:3]
            if 'surface' in l_ana[i]:
                surface = l_ana[i]['surface']
                if surface != text[loc[0]: loc[1]]:
                    logging.info('tagme sf [%s] != [%s]', text[loc[0]:loc[1]], surface)
                    min_st = max(loc[0], min_st)
                    p = text[min_st:].find(surface)
                    if -1 == p:
                        logging.warn('[%s] not in text [%s]', surface, text[min_st:])
                        continue
                    new_st = p + min_st
                    new_ed = new_st + len(surface)
                    old_loc = loc[0], loc[1]
                    loc = (new_st, new_ed)
                    logging.info('moved to [%s] %s -> %s', text[loc[0]:loc[1]],
                                 json.dumps(old_loc), json.dumps(loc))
            min_st = loc[0]

            st, ed = h_char_to_token_loc[loc[0]], h_char_to_token_loc[loc[1] - 1]
            ed += 1
            if 'loc' in l_ana[i]:
                l_ana[i]['loc'] = (st, ed)
            else:
                l_ana[i][1], l_ana[i][2] = st, ed
            before_name = text[loc[0]: loc[1]]
            after_name = ' '.join(text.split()[st:ed])
            if after_name != before_name:
                logging.warn('[%s] location match: [%s] -> [%s]',
                             h_info['docno'], before_name, after_name)
                if len(before_name) <= len(after_name) / 2.0:
                    continue
            l_new_ana.append(l_ana[i])
        if l_ana:
            if 'tagme' in h_info:
                h_info['tagme'][field] = l_new_ana
            else:
                h_info['spot'][field] = l_new_ana
            if len(l_ana) != len(l_new_ana):
                logging.info('[%d] ana to [%d]', len(l_ana), len(l_new_ana))
    return h_info

if __name__ == '__main__':
    from traitlets.config import Configurable
    from traitlets import (
        Unicode,
        List,
    )
    from knowledge4ir.utils import load_py_config, set_basic_log
    set_basic_log()

    class OffsetConvertPara(Configurable):
        in_name = Unicode(help='tagme json results').tag(config=True)
        out_name = Unicode(help='out name').tag(config=True)
        spot_field = Unicode(SPOT_FIELD, help='boe fields: spot|tagme').tag(config=True)
        l_target_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS,
                               help='target fields to convert').tag(config=True)


    if 2 != len(sys.argv):
        print "convert offset from char to token in tagme's ana"
        print "1 para: config"
        OffsetConvertPara.class_print_help()
        sys.exit(-1)
    para = OffsetConvertPara(config=load_py_config(sys.argv[1]))
    out = open(para.out_name, 'w')
    for p, line in enumerate(open(para.in_name)):
        if not p % 100:
            print "converted [%d] lines" % p
        h = json.loads(line)
        h = convert_offset(h, para)
        print >> out, json.dumps(h)
    out.close()
    print "done"
