"""
string replace config
"""

import json
import sys
import logging
from traitlets.config import Configurable
from traitlets import Unicode, List


class SplitConf(Configurable):
    in_name = Unicode(help='in put conf').tag(config=True)
    out_pre = Unicode(help='output conf prefix').tag(config=True)
    place_holder = Unicode('##', help='to replace string').tag(config=True)
    l_target_str = List(Unicode,
                        default_value=["01","02","03","04","05","06","07","08","09","10",
                                       "11","12","13","14","15","16","17","18","19","20",
                                       "21","22","23","24","25","26","27","28","29","30"]
                        ).tag(config=True)


    def process(self):
        lines = open(self.in_name).read().splitlines()
        for suf in self.l_target_str:
            out = open(self.out_pre + '.' + suf, 'w')
            new_lines = [line.replace(self.place_holder, suf) for line in lines]
            print >> out, '\n'.join(new_lines)
            out.close()
            print "[%s] done" % (self.out_pre + '.' + suf)

if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config
    if 2 != len(sys.argv):
        print "split conf, 1 para: config"
        SplitConf.class_print_help()
        sys.exit(-1)

    conf_spliter = SplitConf(config=load_py_config(sys.argv[1]))
    conf_spliter.process()

