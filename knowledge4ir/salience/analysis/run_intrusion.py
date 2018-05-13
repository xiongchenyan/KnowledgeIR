from __future__ import print_function
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)
from knowledge4ir.salience.joint_center import JointSalienceModelCenter
import sys
import logging
import os
import json
from sklearn.metrics import roc_auc_score


def compute_metric(raw_test_docs):
    l_sa_output = []

    for test_doc in raw_test_docs:
        salience_labels = test_doc['event']['bodyText']['salience']
        parsed_test, parsed_gold = predictor.io_parser.parse_data(
            [json.dumps(test_doc)]
        )

        for output in predictor.model(parsed_test):
            if output is None:
                l_sa_output.append([])
            else:
                l_sa_output.append(output.cpu()[0].data.numpy())

        event_index = predictor.output_names.index('event')
        output = l_sa_output[event_index]

        print(output, len(output))
        print(salience_labels, len(salience_labels))

        auc = roc_auc_score(salience_labels, output)
        print(auc)

        sys.stdin.readline()


def intrusion_test(test_path, study_out):
    if not os.path.exists(study_out):
        os.makedirs(study_out)

    progress = 0
    with open(test_path) as test_data:
        for line in test_data:
            tests = json.loads(line)
            sa_tests = tests['salient_test']
            non_sa_tests = tests['non_salient_test']

            compute_metric(sa_tests)
            compute_metric(non_sa_tests)

            progress += 1
            if progress % 10 == 0:
                print(' %d,' % progress, end='')

    print()


if __name__ == '__main__':
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )

    print("Script started")


    class Main(Configurable):
        test_in = Unicode(help='testing data').tag(config=True)
        model_in = Unicode(help='model to read from').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        study_out = Unicode(help='output dir of the study result').tag(
            config=True)
        event_id_pickle = Unicode(help='Event id pickle').tag(config=True)
        entity_id_pickle = Unicode(help='Entity id pickle').tag(config=True)
        entity_emb_in = Unicode(help='Entity embedding').tag(config=True)
        event_emb_in = Unicode(help='Event embedding').tag(config=True)


    if 2 != len(sys.argv):
        print(
            "[Usage] [this script] [config] "
        )

        JointSalienceModelCenter.class_print_help()
        Main.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])

    para = Main(config=conf)
    set_basic_log(logging.getLevelName(para.log_level))

    predictor = JointSalienceModelCenter(config=conf)

    model_loaded = False
    logging.info('Trying to load existing model.')
    if os.path.exists(para.model_in):
        predictor.load_model(para.model_in)
        model_loaded = True
    else:
        logging.info("Cannot find model [%s], "
                     "please set exact path." % para.model_in)
        exit(1)

    intrusion_test(para.test_in, para.study_out)
