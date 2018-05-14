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
    l_model_sa_auc = []

    l_freq_auc = []
    l_freq_sa_auc = []

    event_index = predictor.output_names.index('event')

    for test_doc in raw_test_docs:
        # print("Processing ", test_doc['origin'], test_doc['intruder'])

        origin_saliences = test_doc['event']['bodyText']['origin_salience']

        parsed_test, parsed_gold = predictor.io_parser.parse_data(
            [json.dumps(test_doc)]
        )

        event_gold = parsed_gold[event_index].cpu().data[0].numpy()

        event_gold = [1 if v == 1 else 0 for v in event_gold]

        num_origin = len(origin_saliences)
        num_salient = sum(origin_saliences)

        sa_event_gold = [1] * num_salient + [0] * (len(event_gold) - num_origin)

        ts_evm_feats = parsed_test['ts_evm_feature'].cpu().data[0].numpy()
        freq_feats = ts_evm_feats[:, 6]

        l_sa_output = []
        for output in predictor.model(parsed_test):
            if output is None:
                l_sa_output.append([])
            else:
                l_sa_output.append(output.cpu()[0].data.numpy())

        output = l_sa_output[event_index]

        # Select output by salience.
        osa_indices = [i for (i, s) in enumerate(origin_saliences) if s == 1]
        selection = osa_indices + range(num_origin, len(event_gold))
        selected_output = [output[i] for i in selection]
        selected_freq = [freq_feats[i] for i in selection]

        auc = roc_auc_score(event_gold, output)
        sa_auc = roc_auc_score(sa_event_gold, selected_output)

        freq_auc = roc_auc_score(event_gold, freq_feats)
        sa_freq_auc = roc_auc_score(sa_event_gold, selected_freq)

        l_model_auc.append(auc)
        l_model_sa_auc.append(sa_auc)

        l_freq_auc.append(freq_auc)
        l_freq_sa_auc.append(sa_freq_auc)

    return {
        'model_auc': l_model_auc,
        'freq_auc': l_freq_auc,
        'model_sa_auc': l_model_sa_auc,
        'freq_sa_auc': l_freq_sa_auc,
    }


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

    sa_results = {
        'model_auc': [],
        'freq_auc': [],
        'model_sa_auc': [],
        'freq_sa_auc': [],
    }

    non_sa_results = {
        'model_auc': [],
        'freq_auc': [],
        'model_sa_auc': [],
        'freq_sa_auc': [],
    }

    sa_outs = {}
    for key in sa_results:
        sa_outs[key] = open(os.path.join(study_out, 'salience_test_' + key),
                            'w')

    nsa_outs = {}
    for key in non_sa_results:
        nsa_outs[key] = open(
            os.path.join(study_out, 'non_salience_test_' + key), 'w')

    progress = 0

    with open(test_path) as test_data:
        for line in test_data:
            tests = json.loads(line)
            sa_tests = tests['salient_test']
            non_sa_tests = tests['non_salient_test']

            sa_test_res = run_tests(sa_tests)
            for key, value in sa_test_res.items():
                sa_results[key] = accumulate(sa_results[key], histo(value))
                sa_outs[key].write(tab_list(histo(value)))

            non_sa_test_res = run_tests(non_sa_tests)
            for key, value in non_sa_test_res.items():
                non_sa_results[key] = accumulate(non_sa_results[key],
                                                 histo(value))
                nsa_outs[key].write(tab_list(histo(value)))

            progress += 1
            if progress % 10 == 0:
                print(' %d,' % progress, end='')

    print()

    for key, value in sa_results.items():
        sa_outs[key].write(tab_list(multiply_list(value, 1.0 / progress)))
        sa_outs[key].close()

    for key, value in non_sa_results.items():
        nsa_outs[key].write(tab_list(multiply_list(value, 1.0 / progress)))
        nsa_outs[key].close()


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
