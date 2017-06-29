"""
submit multiple kfold cv for combination of feature files and model files
input:
    list of feature names
    list of model configs
    feature dir
    model dir
    out dir
do:
    submit job for each feature-model conf pair
"""


import logging
import json
import sys
import os
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
    Int,
)
from knowledge4ir.utils.condor import qsub_job


class RunMulKFold(Configurable):
    l_feature_names = List(Unicode, help='list of feature files').tag(config=True)
    l_model_configs = List(Unicode, help='list of model configs').tag(config=True)
    feature_dir = Unicode(help='the dir of features').tag(config=True)
    model_dir = Unicode('.', help='the dir of configs').tag(config=True)
    out_dir = Unicode(help='the output root directory').tag(config=True)

    def submit(self):
        for feature_name in self.l_feature_names:
            for model_config in self.l_model_configs:
                logging.info('submitting %s-%s', feature_name, model_config)
                out_name = feature_name + '.' + model_config
                l_cmd = ['python', 'kfold_pipe.py',
                         os.path.join(self.feature_dir, feature_name),
                         os.path.join(self.model_dir, model_config),
                         os.path.join(self.out_dir, out_name)
                         ]
                qsub_job(l_cmd)
        logging.info('all submitted')
        return


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log, load_py_config
    set_basic_log()
    if 2 != len(sys.argv):
        print "multiple submit kfold jobs"
        print '1 para: config'
        RunMulKFold.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    runner = RunMulKFold(config=conf)
    runner.submit()






