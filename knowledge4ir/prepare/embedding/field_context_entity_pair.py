"""
dump field based entity pairs
input:
    annotated json info
    field a -> b pairs
output:
    each line a context pair

"""

from traitlets.config import Configurable
from traitlets import Unicode
import json
import logging


class FieldContext(Configurable):
    source_field = Unicode(help='source field').tag(config=True)
    target_field = Unicode(help='target field').tag(config=True)
    in_name = Unicode(help='doc info in').tag(config=True)
    out_name = Unicode(help='entity pair out').tag(config=True)

    def process(self):
        out = open(self.out_name, 'w')
        pair_cnt = 0
        for p, line in enumerate(open(self.in_name)):
            if not p % 1000:
                logging.info('[%d] doc [%d] pair', p, pair_cnt)

            h = json.loads(line)
            h_ana = h.get('spot', {})
            l_source_ana = h_ana.get(self.source_field, [])
            l_target_ana = h_ana.get(self.target_field, [])
            pair_cnt += len(l_source_ana) * len(l_target_ana)
            l_s_e = [ana['entities'][0]['id'] for ana in l_source_ana]
            l_t_e = [ana['entities'][0]['id'] for ana in l_target_ana]
            for s_e in l_s_e:
                for t_e in l_t_e:
                    print >> out, s_e + ' ' + t_e

        out.close()
        logging.info('finished with [%d] pair', pair_cnt)


if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config, set_basic_log
    import sys
    set_basic_log()
    if 2 != len(sys.argv):
        print "dump field context"
        print "1 para"
        FieldContext.class_print_help()
        sys.exit(-1)

    runner = FieldContext(config=load_py_config(sys.argv[1]))
    runner.process()

