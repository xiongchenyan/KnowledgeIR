"""
predict data using given model
"""
import logging

from traitlets import (
    Unicode
)
from traitlets.config import Configurable

from knowledge4ir.salience.align_res import AlignPredicted
from knowledge4ir.salience.center import SalienceModelCenter

if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )


    class Main(Configurable):
        test_in = Unicode(help='test in').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        raw_corpus_in = Unicode(help='corpus to align').tag(config=True)
        aligned_corpus_out = Unicode(help='aligned corpus output').tag(config=True)


    if 2 > len(sys.argv):
        print "unit test model train test"
        print "1 para, config with aligning config (optional, set if want to align to raw corpus)"
        SalienceModelCenter.class_print_help()
        Main.class_print_help()
        AlignPredicted.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    para = Main(config=conf)

    set_basic_log(logging.getLevelName(para.log_level))

    model = SalienceModelCenter(config=conf)
    model.load_model(para.model_out)
    model.predict(para.test_in, para.test_out)
    converter = AlignPredicted(config=conf)
    if converter.entity_id_pickle_in:
        logging.info('aligning to [%s]', para.raw_corpus_in)
        converter.align_predict_to_corpus(
            para.raw_corpus_in, para.test_out, para.aligned_corpus_out
        )
        logging.info('alignment finished')

