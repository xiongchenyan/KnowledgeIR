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
from knowledge4ir.salience.utils.evaluation import histo


def run_tests(raw_test_docs):
    l_model_auc = []
    l_freq_auc = []

    event_index = predictor.output_names.index('event')

    for test_doc in raw_test_docs:
        # print("Processing ", test_doc['origin'], test_doc['intruder'])
        parsed_test, parsed_gold = predictor.io_parser.parse_data(
            [json.dumps(test_doc)]
        )

        event_gold = parsed_gold[event_index].cpu().data[0].numpy()
        event_gold = [1 if v == 1 else 0 for v in event_gold]

        ts_evm_feats = parsed_test['ts_evm_feature'].cpu().data[0].numpy()
        freq_feats = ts_evm_feats[:, 6]

        l_sa_output = []
        for output in predictor.model(parsed_test):
            if output is None:
                l_sa_output.append([])
            else:
                l_sa_output.append(output.cpu()[0].data.numpy())

        output = l_sa_output[event_index]

        auc = roc_auc_score(event_gold, output)
        freq_auc = roc_auc_score(event_gold, freq_feats)

        l_model_auc.append(auc)
        l_freq_auc.append(freq_auc)
    return l_model_auc, l_freq_auc


def accumulate(l_all, l):
    if len(l_all) == 0:
        return l
    else:
        return [a + b for a, b in zip(l_all, l)]


def multiply_list(l, multiplier):
    return [e * multiplier for e in l]


def tab_list(l):
    return '\t'.join(["%.6f" % v for v in l]) + '\n'


def intrusion_test(test_path, study_out):
    if not os.path.exists(study_out):
        os.makedirs(study_out)

    results = {
        'salience_model_auc': [],
        'salience_freq_auc': [],
        'non_salience_model_auc': [],
        'non_salience_freq_auc': [],
    }

    outs = {}
    for key in results:
        outs[key] = open(os.path.join(study_out, key), 'w')

    progress = 0

    with open(test_path) as test_data:
        for line in test_data:
            tests = json.loads(line)
            sa_tests = tests['salient_test']
            non_sa_tests = tests['non_salient_test']

            sa_aucs, sa_freq_aucs = run_tests(sa_tests)
            non_sa_aucs, non_sa_freq_aucs = run_tests(non_sa_tests)

            key = 'salience_model_auc'
            res = sa_aucs
            results[key] = accumulate(
                results[key], histo(res)
            )
            outs[key].write(tab_list(histo(res)))

            key = 'salience_freq_auc'
            res = sa_freq_aucs
            results[key] = accumulate(
                results[key], histo(res)
            )
            outs[key].write(tab_list(histo(res)))

            key = 'non_salience_model_auc'
            res = non_sa_aucs
            results[key] = accumulate(
                results[key], histo(res)
            )
            outs[key].write(tab_list(histo(res)))

            key = 'non_salience_freq_auc'
            res = non_sa_freq_aucs
            results[key] = accumulate(
                results[key], histo(res)
            )
            outs[key].write(tab_list(histo(res)))

            progress += 1
            if progress % 10 == 0:
                print(' %d,' % progress, end='')

    print()

    for key in results:
        multiply_list(results[key], 1.0 / progress)
        outs[key].write(results[key])
        outs[key].close()


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
