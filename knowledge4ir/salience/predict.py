"""
predict data using given model
"""

from knowledge4ir.salience.center import SalienceModelCenter
import logging
import json
import sys
from traitlets.config import Configurable
from traitlets import (
    Unicode
)
import torch


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )


    class Main(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)


    if 2 != len(sys.argv):
        print "unit test model train test"
        print "1 para, config"
        SalienceModelCenter.class_print_help()
        Main.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    para = Main(config=conf)

    set_basic_log(logging.getLevelName(para.log_level))

    model = SalienceModelCenter(config=conf)
    model.train(para.train_in, para.valid_in, para.model_out)
    model.load_model(para.model_out)
    model.predict(para.test_in, para.test_out)
