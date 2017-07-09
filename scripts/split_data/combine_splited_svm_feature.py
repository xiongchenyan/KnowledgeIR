"""
combine split svm feature
do:
    cat split svm lines together
    *** assert whether they have the same feature set

"""

from traitlets.config import Configurable
from traitlets import (
    List,
    Unicode,
    Int
)
import json
import logging


class CombineSVMFeature(Configurable):
    l_suffix = List(Unicode,
                    default_value=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                                   "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                   "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]
                    ).tag(config=True)
    l_cw09_p = List(Int, default_value=range(20)).tag(config=True)
    l_cw12_p = List(Int, default_value=range(20, 30)).tag(config=True)
    feature_name_suffix = Unicode('_name.json').tag(config=True)
    cw09_suffix = Unicode('.cw09')
    cw12_suffix = Unicode('.cw12')

    def process(self, in_pre, out_pre):
        l_cw09_in_name = [in_pre + self.l_suffix[p]
                          for p in self.l_cw09_p]
        l_cw12_in_name = [in_pre + self.l_suffix[p]
                          for p in self.l_cw12_p]

        cw09_out_name = out_pre + self.cw09_suffix
        cw12_out_name = out_pre + self.cw12_suffix

        self._check_feature_name(l_cw09_in_name)
        self._check_feature_name(l_cw12_in_name)

        self._combine(l_cw09_in_name, cw09_out_name)
        self._combine(l_cw12_in_name, cw12_out_name)
        logging.info('combined')

    def _check_feature_name(self, l_svm_name):
        l_f_name = [name + self.feature_name_suffix for name in l_svm_name]
        l_h_feature = [json.load(open(f_name)) for f_name in l_f_name]
        h_name = l_h_feature[0]
        for h_feature in l_h_feature[1:]:
            assert h_name == h_feature

    def _combine(self, l_svm_in, out_name):
        lines = sum([open(svm_in).read().splitlines() for svm_in in l_svm_in],
                    [])
        out = open(out_name, 'w')
        print >> out, '\n'.join(lines)
        h = json.load(open(l_svm_in[0] + self.feature_name_suffix))
        json.dump(h, open(out_name + self.feature_name_suffix, 'w'), indent=1)
        return


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import load_py_config, set_basic_log
    set_basic_log()
    if 3 > len(sys.argv):
        print "combine svm"
        print "2+ para: in prefix + out prefix + config (can be default)"
        CombineSVMFeature.class_print_help()
        sys.exit(-1)

    if len(sys.argv) > 3:
        combiner = CombineSVMFeature(config=load_py_config(sys.argv[3]))
    else:
        combiner = CombineSVMFeature()
    combiner.process(sys.argv[1], sys.argv[2])
