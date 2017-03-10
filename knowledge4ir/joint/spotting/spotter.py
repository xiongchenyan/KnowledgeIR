"""
the spotter
keep all top candidate
uses the sf dict in the resource class: JointSemanticResource()
    so as to share the resource in memory
add APi to run q_info and d_info's json format
"""

import sys
import json
from knowledge4ir.joint.resource import JointSemanticResource
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode,
    List,
    Bool
)
from knowledge4ir.utils import (
    set_basic_log,
    load_py_config
)
import logging
from copy import deepcopy
from knowledge4ir.utils import TARGET_TEXT_FIELDS


class Spotter(Configurable):
    max_surface_len = Int(5, help='max surface form length').tag(config=True)
    max_candidate_per_surface = Int(5, help='max candidate per surface').tag(config=True)
    only_longest = Bool(False, help='whether only keep longest').tag(config=True)

    def __init__(self, **kwargs):
        super(Spotter, self).__init__(**kwargs)
        self.resource = JointSemanticResource(**kwargs)  # can be initialized here or by set_resource

    @classmethod
    def class_print_help(cls, inst=None):
        super(Spotter, cls).class_print_help()
        JointSemanticResource.class_print_help(inst)

    def set_resource(self, joint_resource):
        # share one resource
        self.resource = joint_resource

    def spot_text(self, l_terms):
        """
        l_terms is the tokenized terms of the text
        :param l_terms:
        :return: l_spot [[surface form, st, ed, [(fb id, p(fb id| sf)),]]]
        """
        logging.info('start spotting text of [%d] terms', len(l_terms))
        l_spot = []
        st = 0
        while st < len(l_terms):
            for reverse_len in xrange(self.max_surface_len):
                ed = st + self.max_surface_len - reverse_len
                if ed > len(l_terms):
                    continue
                sub_str = ' '.join(l_terms[st: ed])

                if len(sub_str) > 3:
                    # manual set capitalization priority
                    l_variation_ngram = [sub_str.title(), sub_str]
                else:
                    l_variation_ngram = [sub_str]
                spotted = False
                for ngram in l_variation_ngram:
                    l_ana = self._get_candidate(ngram)
                    if l_ana:
                        spotted = True
                        # res = [ngram, st, ed, l_ana]
                        res = dict()
                        res['surface'] = ngram
                        res['loc'] = (st, ed)
                        l_entities = [{"id": ana[0], 'cmns': ana[1]} for ana in l_ana]
                        res['entities'] = l_entities
                        l_spot.append(res)
                        logging.debug('get spot [%s] in [%d-%d)', ngram, st, ed)
                        break
                if self.only_longest:
                    if spotted:
                        st = ed - 1
                        break
            st += 1

        logging.info('[%d] terms resulted in [%d] surfaces', len(l_terms), len(l_spot))
        return l_spot

    def _get_candidate(self, ngram):
        """
        get possible candidates for ngram
        :param ngram: ngram to tag
        :return: [(fb id, p(fb id | ngram)),]
        """
        h_sf = self.resource.h_surface_form

        l_mid_ana = h_sf.get(ngram, [])
        logging.debug('[%s] [%d] candidate', ngram, len(l_mid_ana))
        l_ana = deepcopy(l_mid_ana[:self.max_candidate_per_surface])
        return l_ana

    def pipe_spot_query_json(self, q_info_in, spot_out_name):
        """
        spot packed json
        :param q_info_in: json query, has the field ['query'] in each line
        :param spot_out_name: add 'spot' field (also split() and join the text)
        :return:
        """
        logging.info('spotting q [%s] to [%s]', q_info_in, spot_out_name)
        q_cnt = 0
        spot_cnt = 0
        out = open(spot_out_name, 'w')
        for line in open(q_info_in):
            h_q = json.loads(line)
            h_spot = self.spot_query_json(h_q)
            q_cnt += 1
            spot_cnt += len(h_spot['spot'])
            print >> out, json.dumps(h_spot)
        out.close()
        logging.info('spotted, average [%d] sf per q', float(spot_cnt) / float(q_cnt))
        return

    def spot_query_json(self, h_q):
        h = dict()
        q = h_q['query']
        l_qt = q.lower().split()
        l_ana = self.spot_text(l_qt)
        h['query'] = ' '.join(l_qt)
        h['spot'] = {'query': l_ana}
        return h

    def pipe_spot_doc_json(self, d_info_in, spot_out_name):
        """

        :param d_info_in: json doc in, had fields of TARGET_TEXT_FIELDS
        :param spot_out_name: add 'spot' field to the json (also split() and join the text)
        :return:
        """

        logging.info('spotting doc [%s] to [%s]', d_info_in, spot_out_name)
        d_cnt = 0
        out = open(spot_out_name, 'w')
        for line in open(d_info_in):
            h_d = json.loads(line)
            h_spot = self.spot_doc_json(h_d)
            d_cnt += 1
            print >> out, json.dumps(h_spot)
        out.close()
        logging.info('spotted [%d] documents', d_cnt)
        return

    def spot_doc_json(self, h_d):
        h = dict()
        h['spot'] = dict()
        for field in TARGET_TEXT_FIELDS:
            if field not in h_d:
                continue
            text = h_d[field]
            l_term = text.lower().split()
            l_ana = self.spot_text(l_term)
            h[field] = ' '.join(l_term)
            h['spot'][field] = l_ana
            h['docno'] = h_d['docno']

        return h


class MainConf(Configurable):
    in_name = Unicode(help='input').tag(config=True)
    out_name = Unicode(help='output').tag(config=True)
    in_type = Unicode('q', help='q|doc').tag(config=True)


if __name__ == '__main__':
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "I spot q or doc"
        print "conf:"
        Spotter.class_print_help()
        MainConf.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])
    arg = MainConf(config=conf)

    spotter = Spotter(config=conf)
    if arg.in_type == 'q':
        spotter.pipe_spot_query_json(arg.in_name, arg.out_name)
    else:
        spotter.pipe_spot_doc_json(arg.in_name, arg.out_name)



