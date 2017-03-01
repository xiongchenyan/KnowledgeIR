"""
parallel submit extractor or att_ltr_feature_extractor
input:
    doc_info_in dir (to parallel doc in as input)
    out_dir (to parallel out put dir)
    extractor's conf
do:
    submit a job for each doc info in the dir, with matched suffix in the out_dir
"""

import ntpath
import os
import logging
import json

from knowledge4ir.utils.condor import qsub_job
from traitlets.config import Configurable
from traitlets import (
    Unicode,
)


class ParallelExtractor(Configurable):
    extractor_name = Unicode('extractor.py', help='job to run').tag(config=True)
    in_dir = Unicode(help='doc info indir').tag(config=True)
    out_dir = Unicode(help='out dir').tag(config=True)

    def __init__(self, **kwargs):
        super(ParallelExtractor, self).__init__(**kwargs)
        self.base_conf_in = ""
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def _form_in_out(self, base_conf_in):
        base_out_line = ""
        for line in open(base_conf_in):
            if 'out_name=' in line:
                base_out_line = line.strip()
                break
        if not base_out_line:
            logging.warn('base out missing')
        base_out_name = base_out_line.split('=')[-1].strip('"')
        base_out_name = ntpath.basename(base_out_name)
        l_in_out_names = []

        for dir_name, sub_dirs, fnames in os.walk(self.in_dir):
            for fname in fnames:
                in_name = os.path.join(dir_name, fname)
                out_name = os.path.join(self.out_dir, base_out_name + '.' + fname)
                l_in_out_names.append([in_name, out_name])
        return l_in_out_names

    def _sub_jobs(self, l_in_out_names):
        for in_name, out_name in l_in_out_names:
            l_cmd = ['python', self.extractor_name, self.base_conf_in, in_name, out_name]
            qsub_job(l_cmd)
        logging.info('[%d] jobs submitted', len(l_in_out_names))

    def sub_jobs(self, base_conf_in):
        self.base_conf_in = base_conf_in
        l_in_out_names = self._form_in_out(self.base_conf_in)
        self._sub_jobs(l_in_out_names)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    set_basic_log()
    if 3 != len(sys.argv):
        print "I submit feature extractors in parallel"
        print "2 para: this conf in + extractor base conf in"
        ParallelExtractor.class_print_help()
        sys.exit()

    conf = load_py_config(sys.argv[1])
    extractor = ParallelExtractor(config=conf)
    extractor.sub_jobs(sys.argv[2])


