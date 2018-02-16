from traitlets import (
    Unicode,
    Int,
    Float,
    List,
    Bool
)
from traitlets.config import Configurable
import json
import logging
import math
import os
import torch
import numpy as np

from knowledge4ir.salience.joint_center import JointSalienceModelCenter


def merge_pack(h_packed_data_org, h_packed_data_ext):
    h_packed_data_mixed = {}
    for key in h_packed_data_org:
        data_org = h_packed_data_org[key]
        data_ext = h_packed_data_ext[key]

        if key == 'ts_adjacent' or key == 'ts_args':
            continue
        elif key == 'masks':
            h_mixed_masks = {}
            for mask_key in data_org:
                if mask_key == 'ts_args':
                    continue

                mask_org = h_packed_data_org[key][mask_key]
                mask_ext = h_packed_data_ext[key][mask_key]
                mask_mixed = torch.cat((mask_org, mask_ext), 1)
                h_mixed_masks[mask_key] = mask_mixed

            h_packed_data_mixed[key] = h_mixed_masks
        else:
            data_mixed = torch.cat((data_org, data_ext), 1)
            h_packed_data_mixed[key] = data_mixed

    return h_packed_data_mixed


def select_pack(h_packed_data, l_v_label, inds_e, inds_evm):
    h_packed_data_selected = {}

    inds_e_ts = torch.from_numpy(inds_e).cuda()
    inds_evm_ts = torch.from_numpy(inds_evm).cuda()

    num_entities = len(inds_e[0])
    adjacent = h_packed_data['ts_adjacent']
    evm_args = adjacent[:, num_entities:, :]


    evm_keys = {'mtx_evm_score', 'mtx_evm', 'label_evm', 'ts_evm_feature',
                'ts_adjacent', 'mtx_arg_length', 'ts_args'}

    e_keys = {'mtx_e_score', 'ts_e_feature', 'mtx_e', 'label_e'}

    for key, data in h_packed_data.items():
        if key in evm_keys:
            h_packed_data_selected[key] = data[:, inds_evm_ts].squeeze(0)
        elif key in e_keys:
            h_packed_data_selected[key] = data[:, inds_e_ts].squeeze(0)
        elif key == 'masks':
            sorted_masks = {}
            for mask_key, mask in data.items():
                if mask_key in e_keys:
                    sorted_masks[mask_key] = mask[:, inds_e_ts].squeeze(0)
                elif mask_key in evm_keys:
                    sorted_masks[mask_key] = mask[:, inds_evm_ts].squeeze(0)

            h_packed_data_selected[key] = sorted_masks

    l_v_label_selected = [
        l_v_label[0][:, inds_e_ts].squeeze(0),
        l_v_label[1][:, inds_evm_ts].squeeze(0),
    ]

    return h_packed_data_selected, l_v_label_selected


def sort_pack(h_packed_data, l_v_label, reverse):
    # TODO e should be obtained from the adjacent matrix.
    inds_e = l_v_label[0].cpu().data.numpy().argsort()
    inds_evm = l_v_label[1].cpu().data.numpy().argsort()

    if reverse:
        inds_e = inds_e[::-1]
        inds_evm = inds_evm[::-1]

    return select_pack(h_packed_data, l_v_label, inds_e, inds_evm)


def mix(docline1, docline2, io_parser, reverse=False):
    h_packed_data_org, l_v_label_org = io_parser.parse_data([docline1])
    h_packed_data_ext, l_v_label_ext = io_parser.parse_data([docline2])

    h_packed_data_ext, l_v_label_ext = sort_pack(h_packed_data_ext,
                                                 l_v_label_ext, reverse)

    for i in range(l_v_label_ext[1].size()[1]):
        l_v_label_mixed = [torch.cat((v_label_org, v_label_ext), 1) for
                           v_label_org, v_label_ext in
                           zip(l_v_label_org, l_v_label_ext)]
        h_packed_data_mixed = merge_pack(h_packed_data_org, h_packed_data_ext)

        yield h_packed_data_mixed, l_v_label_mixed


def kernel_word_counting(model, test_in_path):
    pass


def intrusion_test(predictor, test_in_path, num_tests=1,
                   num_intruder_per=10):
    test_count = 0
    intruder_set_count = 0
    intruder_count = 0

    test_data = [[] for _ in range(num_tests)]

    print 'Loading some test files'
    with open(test_in_path) as test_in:
        for line in test_in:
            if predictor.io_parser.is_empty_line(line):
                continue
            if test_count < num_tests:
                test_data[test_count] = (line, [])
                test_count += 1
            elif intruder_set_count < num_tests:
                test_data[intruder_set_count][1].append(line)
                intruder_count += 1
            else:
                break

            if intruder_count == num_intruder_per:
                intruder_count = 0
                intruder_set_count += 1

    print 'Loaded %d test pairs, each with %d intruders.' % (
        len(test_data), len(test_data[0][1]))

    for origin, intruders in test_data:
        for intruder in intruders:

            for h_packed_data, l_v_label in mix(origin, intruder,
                                                predictor.io_parser):
                l_output = []
                for output in predictor.model(h_packed_data):
                    if output is None:
                        l_output.append([])
                    else:
                        l_output.append(output.cpu()[0])

                for i, name in enumerate(predictor.output_names):
                    output = l_output[i]
                    v_label = l_v_label[i]

                    l_score = output.data.numpy().tolist()
                    l_label = v_label[0].cpu().data.numpy().tolist()

                    print 'predicting mixing results.'
                    print l_score
                    print l_label

                    import sys
                    sys.stdin.readline()


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
        model_in = Unicode(help='model to read from').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        skip_train = Bool(False, help='directly test').tag(config=True)
        debug = Bool(False, help='Debug mode').tag(config=True)


    if 3 != len(sys.argv):
        print "[Usage] [this script] [config] " \
              "[study mode(1=intruder, 2=kernel)]"

        JointSalienceModelCenter.class_print_help()
        Main.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    mode = sys.argv[2]
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

    intrusion_test(predictor, para.test_in)
