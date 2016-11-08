"""
get words from q info and doc info
input:
    list of q info and doc info file names
    target fields
output:
    list of words
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


class GetVocabulary(Configurable):
    l_target_fname = List(Unicode).tag(config=True)
    l_target_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS + ['query']).tag(config=True)
    out_name = Unicode().tag(config=True)

    def _get_per_f(self, fname):
        l_w = []
        logging.info('started [%s]', fname)
        for line in open(fname):
            h = json.loads(line.split('\t')[-1])
            for field in self.l_target_fields:
                if field in h:
                    l_w.extend(list(set(h[field].split())))
        logging.info('[%d] words get from [%s]', len(l_w), fname)
        return l_w

    def get_vocabulary(self):
        l_total = []
        for fname in self.l_target_fname:
            l_w = self._get_per_f(fname)
            l_total.extend(list(set(l_w)))
        l_total = list(set(l_total))
        out = open(self.out_name, 'w')
        for w in l_total:
            print >> out, w
        logging.info('total [%d] size vocabulary', len(l_total))
        return


if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if len(sys.argv) != 2:
        print "1 para: config"
        GetVocabulary.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    getter = GetVocabulary(config=conf)
    getter.get_vocabulary()