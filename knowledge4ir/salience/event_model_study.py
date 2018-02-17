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
from torch.autograd import Variable

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


def select_pack(h_packed_data, l_v_label, selected_args, indices_evm):
    h_packed_data_selected = {}

    inds_e_ts = torch.from_numpy(selected_args).cuda()

    inds_evm_ts = torch.from_numpy(indices_evm).cuda()

    evm_keys = {'mtx_evm_score', 'mtx_evm', 'label_evm', 'ts_evm_feature',
                'ts_adjacent', 'mtx_arg_length', 'ts_args'}

    e_keys = {'mtx_e_score', 'ts_e_feature', 'mtx_e', 'label_e'}

    for key, data in h_packed_data.items():
        if key in evm_keys:
            h_packed_data_selected[key] = data[:, inds_evm_ts]
        elif key in e_keys:
            h_packed_data_selected[key] = data[:, inds_e_ts]
        elif key == 'masks':
            sorted_masks = {}
            for mask_key, mask in data.items():
                if mask_key in e_keys:
                    sorted_masks[mask_key] = mask[:, inds_e_ts]
                elif mask_key in evm_keys:
                    sorted_masks[mask_key] = mask[:, inds_evm_ts]

            h_packed_data_selected[key] = sorted_masks

    l_v_label_selected = [
        l_v_label[0][:, inds_e_ts],
        l_v_label[1][:, inds_evm_ts],
    ]

    return h_packed_data_selected, l_v_label_selected


def select_pack_ordered(h_packed_data, l_v_label, reverse=True):
    # Selected events.
    indices_evm = l_v_label[1].cpu().data.numpy().argsort()[0][::-1]

    if reverse:
        indices_evm = indices_evm[::-1]

    e_labels = l_v_label[0].cpu().data.numpy()[0]
    num_entities = len(e_labels)
    adjacent = h_packed_data['ts_adjacent']

    evm_adjacent = adjacent[0, num_entities:, :]

    endings = range(1, indices_evm.size + 1)
    if reverse:
        endings.reverse()

    for end in endings:
        ts_arg_selector = Variable(torch.zeros(evm_adjacent.size()).cuda())

        for i in indices_evm[:end]:
            ts_arg_selector[i, :] = 1

        selected_adjacent = evm_adjacent * ts_arg_selector

        nonzeros = torch.nonzero(selected_adjacent)

        if not nonzeros.size():
            continue
        else:
            selected_args = np.unique(torch.nonzero(selected_adjacent)[:, 1]
                                      .cpu().data.numpy())

        selected_events = indices_evm[:end].copy()

        yield select_pack(h_packed_data, l_v_label, selected_args,
                          selected_events)


def mix(h_packed_data_1, h_packed_data_2, l_v_label_1, l_v_label_2):
    l_v_label_mixed = [torch.cat((v_label_org, v_label_ext), 1) for
                       v_label_org, v_label_ext in
                       zip(l_v_label_1, l_v_label_2)]
    h_packed_data_mixed = merge_pack(h_packed_data_1,
                                     h_packed_data_2)
    org_size = [v.size()[1] for v in l_v_label_1]
    ext_size = [v.size()[1] for v in l_v_label_2]
    return h_packed_data_mixed, l_v_label_mixed, org_size, ext_size


def mix_all(org_line, ext_line):
    h_packed_data_org, l_v_label_org = org_line
    h_packed_data_ext, l_v_label_ext = ext_line

    for h_packed_data_ext_sel, l_v_label_ext_sel in select_pack_ordered(
            h_packed_data_ext, l_v_label_ext, False):
        yield mix(h_packed_data_org, h_packed_data_ext_sel, l_v_label_org,
                  l_v_label_ext_sel)

    for h_packed_data_org_sel, l_v_label_org_sel in select_pack_ordered(
            h_packed_data_org, l_v_label_org, True):
        yield mix(h_packed_data_org_sel, h_packed_data_ext, l_v_label_org_sel,
                  l_v_label_ext)


def kernel_word_counting(model, test_in_path):
    pass


def analyze_intrusion(l_score, l_label, org_size):
    import operator
    sorted_scores = sorted(zip(l_score, enumerate(l_label)), reverse=True,
                           key=operator.itemgetter(0))

    o_mrr = 0
    e_mrr = 0

    oc = 0
    ec = 0

    for rank, (score, (index, label)) in enumerate(sorted_scores):
        if label == 1:
            if index >= org_size:
                e_mrr += 1.0 / (rank + 1)
                ec += 1
            else:
                o_mrr += 1.0 / (rank + 1)
                oc += 1

    o_mrr = o_mrr / len(l_label)
    e_mrr = e_mrr / len(l_label)

    return o_mrr, e_mrr


def intrusion_test(predictor, test_in_path, num_tests=1,
                   num_intruder_per=10):
    test_count = 0
    intruder_set_count = 0
    intruder_count = 0

    test_threshold = 20

    test_data = [[] for _ in range(num_tests)]

    print 'Loading some test files'
    with open(test_in_path) as test_in:
        for line in test_in:
            if predictor.io_parser.is_empty_line(line):
                continue

            h_packed_data, l_v_label = predictor.io_parser.parse_data([line])

            event_labels = l_v_label[1]
            num_events = event_labels.size()[1]
            if num_events < test_threshold:
                continue

            if not np.any([l == 1 for l in event_labels.cpu().data.numpy()]):
                continue

            if test_count < num_tests:
                test_data[test_count] = ((h_packed_data, l_v_label), [])
                test_count += 1
            elif intruder_set_count < num_tests:
                test_data[intruder_set_count][1].append(
                    (h_packed_data, l_v_label))
                intruder_count += 1
            else:
                break

            if intruder_count == num_intruder_per:
                intruder_count = 0
                intruder_set_count += 1

    print 'Loaded %d test pairs, each with %d intruders.' % (
        len(test_data), len(test_data[0][1]))

    for origin, intruders in test_data:
        origin_labels = origin[1][1].cpu().data.numpy()[0]
        num_origin_positives = sum(
            [1 if l == 1 else 0 for l in origin_labels])
        num_origin = len(origin_labels)

        for intruder in intruders:
            intruder_labels = intruder[1][1].cpu().data.numpy()[0]
            num_intruder_positives = sum(
                [1 if l == 1 else 0 for l in intruder_labels])
            num_intruder = len(intruder_labels)

            l_o_mrr = []
            l_e_mrr = []
            totals = []
            for h_packed_data, l_v_label, org_sizes, \
                ext_sizes in mix_all(origin, intruder):

                l_output = []
                for output in predictor.model(h_packed_data):
                    if output is None:
                        l_output.append([])
                    else:
                        l_output.append(output.cpu()[0])

                for i, name in enumerate(predictor.output_names):
                    if name == 'event':
                        output = l_output[i]
                        v_label = l_v_label[i]

                        org_size = org_sizes[i]

                        l_score = output.data.numpy().tolist()
                        l_label = v_label[0].cpu().data.numpy().tolist()

                        o_mrr, e_mrr = analyze_intrusion(l_score,
                                                         l_label,
                                                         org_size)

                        totals.append(len(l_label))

                        l_o_mrr.append(o_mrr)
                        l_e_mrr.append(e_mrr)

            print "Number of origin %d, number of intruder %d, number of " \
                  "origin positives %d, number of intruder " \
                  "positives %d." % (
                      num_origin, num_intruder, num_origin_positives,
                      num_intruder_positives)

            print totals
            print l_o_mrr
            print l_e_mrr
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
