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
    Unicode
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

    def _get_per_f(self, fname):
        l_e = []
        for line in open(fname):
            h = json.loads(line.split('\t')[-1])
            for linker in self.l_linker:
                if linker not in h:
                    continue
                for field in self.l_target_fields:
                    if field in h:
                        l_e.extend([ana[0] for ana in h[field]])
        logging.info('[%d] candidate get from [%s]'. len(l_e), fname)
        return l_e

    def get_candidate_e(self):
        s_e = set()
        for fname in self.l_target_fname:
            l_e = self._get_per_f(fname)
            s_e.update(set(l_e))
        print >> open(self.out_name,'w'), '\n'.join(list(s_e))
        logging.info('total [%d] candidate entities', len(s_e))
        return


if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config
    if len(sys.argv) != 2:
        print "1 para: config"
        GetCandidateEntity.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    getter = GetCandidateEntity(config=conf)
    getter.get_candidate_e()







