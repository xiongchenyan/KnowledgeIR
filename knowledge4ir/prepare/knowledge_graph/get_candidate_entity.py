"""
get candidate entity from a list of file name
input:
    q info and doc info's file names
    target fields (query, bodyText, title)
output:
    candidate entities
"""

from traitlets.config import Configurable
from traitlets import (
    List,
    Unicode,
    Bool,
)
import json
import sys
import logging
from knowledge4ir.utils import TARGET_TEXT_FIELDS


class GetCandidateEntity(Configurable):
    l_target_fname = List(Unicode).tag(config=True)
    l_target_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS + ['query']).tag(config=True)
    l_linker = List(Unicode, default_value=['tagme', 'cmns']).tag(config=True)
    out_name = Unicode().tag(config=True)
    with_tf = Bool(False).tag(config=True)

    def _get_per_f(self, fname):
        l_e = []
        logging.info('started [%s]', fname)
        for line in open(fname):
            h = json.loads(line.split('\t')[-1])
            for linker in self.l_linker:
                if linker not in h:
                    continue
                for field in self.l_target_fields:
                    if field in h[linker]:
                        l_e.extend(list(set([ana['entities']['id'] for ana in h[linker][field]])))
        logging.info('[%d] candidate get from [%s]', len(l_e), fname)
        return l_e

    def get_candidate_e(self):
        h_tf = dict()
        l_total = []
        for fname in self.l_target_fname:
            l_e = self._get_per_f(fname)
            if self.with_tf:
                for e in l_e:
                    h_tf[e] = h_tf.get(e, 0) + 1
            else:
                l_total.extend(list(set(l_e)))
        out = open(self.out_name, 'w')

        if self.with_tf:
            l_e_tf = h_tf.items()
            l_e_tf.sort(key=lambda item: item[1], reverse=True)
            for e, tf in l_e_tf:
                print >> out, e + '\t%d' % tf
        else:
            l_total = list(set(l_total))
            for e in l_total:
                print >> out, e
        out.close()
        logging.info('total [%d] candidate entities', len(l_total))
        return


if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if len(sys.argv) != 2:
        print "1 para: config"
        GetCandidateEntity.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    getter = GetCandidateEntity(config=conf)
    getter.get_candidate_e()







