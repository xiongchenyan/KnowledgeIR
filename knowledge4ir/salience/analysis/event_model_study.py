from __future__ import print_function

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from traitlets import (
    Unicode,
    Int
)
from traitlets.config import Configurable
import math
from knowledge4ir.salience.joint_center import JointSalienceModelCenter
import sys
import operator
from sklearn.metrics import roc_auc_score
from collections import Counter


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


def __recover_events(mtx_feature):
    l_features = mtx_feature.cpu().data.numpy()[0]

    groups = []
    group_ids = []

    def pick(f):
        picked = f[:]
        picked[7] = 0
        picked[10] = 0
        picked[11] = 0
        picked[12] = 0
        return picked

    for features in l_features:
        picked_feature = pick(features)
        for gid, existing in enumerate(groups):
            if np.array_equal(existing, picked_feature):
                group_ids.append(gid)
                break
        else:
            groups.append(picked_feature)
            group_ids.append(len(group_ids))

    return group_ids


def __filter_features(mtx_feature, event_selector, events):
    rows = event_selector.cpu().numpy()
    l_features = mtx_feature.cpu().data.numpy()[0]

    event_counts = Counter()
    for r in rows:
        event_counts[events[r]] += 1

    for r in rows:
        event_id = events[r]
        count = event_counts[event_id]
        l_features[r][6] = count
        # for k in range(7, 13):
        #     l_features[r][k] = 0

    masked_features = l_features[event_selector]

    ext_masked_res = Variable(
        torch.from_numpy(masked_features).unsqueeze(0).cuda()
    )

    return ext_masked_res


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


def select_pack_ordered(h_packed_data, l_v_label, good_first=True):
    # Sort the event using the labels.
    if good_first:
        indices_evm = l_v_label[1].cpu().data.numpy().argsort()[0][::-1]
    else:
        indices_evm = l_v_label[1].cpu().data.numpy().argsort()[0]

    evm_labels = [1 if v == 1 else 0 for v in
                  l_v_label[1].cpu().data.numpy()[0]]
    num_positives = sum(evm_labels)

    e_labels = l_v_label[0].cpu().data.numpy()[0]
    num_entities = len(e_labels)
    adjacent = h_packed_data['ts_adjacent']

    evm_adjacent = adjacent[0, num_entities:, :]

    assert num_positives * 5 <= indices_evm.size

    # We add at most 5x non-salient events to see the outcome.
    if good_first:
        endings = range(1, num_positives * 5 + 1)
        begins = [0] * len(endings)
        selection_range = zip(begins, endings)
    else:
        total = len(indices_evm)
        cutoff = max(0, total - num_positives * 5)

        endings = range(cutoff + 1, total + 1)

        begins = [cutoff] * len(endings)
        selection_range = zip(begins, endings)

    count = 0
    for begin, end in selection_range:
        # Need to select subset from the adjacent matrix.
        ts_arg_selector = Variable(torch.zeros(evm_adjacent.size()).cuda())
        for i in indices_evm[begin:end]:
            ts_arg_selector[i, :] = 1
        selected_adjacent = evm_adjacent * ts_arg_selector
        nonzeros = torch.nonzero(selected_adjacent)

        if not nonzeros.size():
            continue
        else:
            selected_args = np.unique(torch.nonzero(selected_adjacent)[:, 1]
                                      .cpu().data.numpy())

        selected_events = indices_evm[begin:end].copy()

        count += 1
        yield select_pack(h_packed_data, l_v_label, selected_args,
                          selected_events)


def mix(h_packed_data_org, h_packed_data_ext, l_v_label_org, l_v_label_ext):
    l_v_label_mixed = [torch.cat((v_label_org, v_label_ext), 1) for
                       v_label_org, v_label_ext in
                       zip(l_v_label_org, l_v_label_ext)]
    h_packed_data_mixed = merge_pack(h_packed_data_org,
                                     h_packed_data_ext)
    org_size = [v.size()[1] for v in l_v_label_org]
    ext_size = [v.size()[1] for v in l_v_label_ext]

    return h_packed_data_mixed, l_v_label_mixed, org_size, ext_size


def mix_with_preference(org_line, ext_line, good_first=True):
    h_packed_data_org, l_v_label_org = org_line
    h_packed_data_ext, l_v_label_ext = ext_line

    for h_packed_data_ext_sel, l_v_label_ext_sel in select_pack_ordered(
            h_packed_data_ext, l_v_label_ext, good_first):
        yield mix(h_packed_data_org, h_packed_data_ext_sel, l_v_label_org,
                  l_v_label_ext_sel)


def analyze_intrusion(l_score, l_label, org_size, ext_size):
    sorted_scores = sorted(
        zip(l_score, enumerate(l_label)), reverse=True,
        key=operator.itemgetter(0)
    )

    total_origin_sa = sum([1 if v == 1 else 0 for v in l_label[:org_size]])

    origin_lowers = [0] * ext_size
    origin_salience_lowers = [0] * ext_size

    num_origin_seen = 0
    num_sa_origin_seen = 0

    num_intruder = 0
    num_intruder_sa = 0

    org_sa_scores = []

    for rank, (score, (index, label)) in enumerate(sorted_scores):
        if index >= org_size:
            # These are intruders.
            origin_lowers[index - org_size] = (
                label, org_size - num_origin_seen)
            origin_salience_lowers[index - org_size] = (
                label, total_origin_sa - num_sa_origin_seen)
            num_intruder += 1
            if label == 1:
                num_intruder_sa += 1
        else:
            # These are origins.
            if label == 1:
                num_sa_origin_seen += 1
                org_sa_scores.append(score)
            num_origin_seen += 1

    aver_rank = 0
    sa_aver_rank = 0

    aver_rank_in_sa = 0
    sa_aver_rank_in_sa = 0

    for label, num_origin_lower in origin_lowers:
        aver_rank += 1.0 * num_origin_lower / org_size
        if label == 1:
            sa_aver_rank += 1.0 * num_origin_lower / org_size

    for label, num_origin_s_lower in origin_salience_lowers:
        aver_rank_in_sa += 1.0 * num_origin_s_lower / total_origin_sa
        if label == 1:
            sa_aver_rank_in_sa += 1.0 * num_origin_s_lower / total_origin_sa

    aver_rank /= num_intruder
    aver_rank_in_sa /= num_intruder

    if num_intruder_sa:
        sa_aver_rank /= num_intruder_sa
        sa_aver_rank_in_sa /= num_intruder_sa
    else:
        sa_aver_rank_in_sa = 0
        sa_aver_rank = 0

    org_all_scores = l_score[:org_size]
    ext_all_scores = l_score[org_size:]

    auc = __intruder_auc(org_all_scores, ext_all_scores)

    # AUC score among salient origins.
    sa_auc = __intruder_auc(org_sa_scores, ext_all_scores)

    results = {
        'average_rank': aver_rank,
        'average_rank_in_salience': aver_rank_in_sa,
        'average_salience_rank': sa_aver_rank,
        'average_salience_rank_in_salience': sa_aver_rank_in_sa,
        'auc': auc,
        'salience_auc': sa_auc,
    }

    return results


def __load_intruder_data(test_in_path, num_tests, num_intruder_per):
    test_data = [[] for _ in range(num_tests)]

    test_count = 0
    intruder_set_count = 0
    intruder_count = 0

    test_threshold = 20

    print('Loading some test files')
    with open(test_in_path) as test_in:
        for line in test_in:
            if predictor.io_parser.is_empty_line(line):
                continue

            h_packed_data, l_v_label = predictor.io_parser.parse_data([line])

            event_labels = l_v_label[1]

            if event_labels is None:
                continue

            num_events = event_labels.size()[1]
            if num_events < test_threshold:
                continue

            num_salient = sum(
                [1 if l == 1 else 0 for l in
                 event_labels.cpu().data.numpy()[0]])

            if num_salient < 5:
                continue

            if num_salient * 5 > num_events:
                continue

            if test_count < num_tests:
                # test_data[test_count] = ((h_packed_data, l_v_label), [])
                test_data[test_count] = (line, [])
                test_count += 1
            elif intruder_set_count < num_tests:
                # test_data[intruder_set_count][1].append(
                #     (h_packed_data, l_v_label))
                test_data[intruder_set_count][1].append(line)
                intruder_count += 1
            else:
                break

            if intruder_count == num_intruder_per:
                intruder_count = 0
                intruder_set_count += 1

    print('Loaded %d test pairs, each with %d intruders.' % (
        len(test_data), len(test_data[0][1])))

    return test_data


def __collect_intruder_result(test_data, good_first, out_dir):
    histograms = {
        'average_rank': [],
        'average_rank_in_salience': [],
        'average_salience_rank': [],
        'average_salience_rank_in_salience': [],
        'auc': [],
        'salience_auc': [],
        'leading_auc': [],
        'leading_salience_auc': [],
    }

    order = 'sa_first' if good_first else 'sa_last'

    outputs = dict(
        [(k, open(os.path.join(out_dir, '%s.%s.tsv' % (k, order)), 'w')) for k
         in histograms.keys()])

    progress = 0
    total_pairs = 0

    print("Processed:", end='')
    for origin_line, intruder_lines in test_data:
        origin = predictor.io_parser.parse_data([origin_line])

        progress += 1
        if progress % 10 == 0:
            print(' %d,' % progress, end='')

        for intruder_line in intruder_lines:
            total_pairs += 1

            intruder = predictor.io_parser.parse_data([intruder_line])

            results = test_intruder(
                mix_with_preference(origin, intruder, good_first)
            )

            for key, l_res in results.items():
                res_histo = histo(l_res)
                histograms[key] = accumulate(
                    histograms[key], res_histo
                )
                outputs[key].write(tab_list(res_histo))

    print()

    m = 1.0 / total_pairs

    for key, h in histograms.items():
        aver_h = multiply_list(h, m)
        outputs[key].write("Final line shows average:\n")
        outputs[key].write(tab_list(aver_h))

    for out in outputs.values():
        out.close()

    print('Number of paris processed : %d' % total_pairs)
    print('Ordering is good first: %s' % good_first)


def intrusion_test(test_in_path, out_dir, num_tests=1, num_intruder_per=10):
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_data = __load_intruder_data(test_in_path, num_tests, num_intruder_per)

    print("Processing test data from salience first")
    __collect_intruder_result(test_data, True, out_dir)

    print("Processing test data from non-salience first")
    __collect_intruder_result(test_data, False, out_dir)


def tab_list(l):
    return '\t'.join(["%.6f" % v for v in l]) + '\n'


def accumulate(l_all, l):
    if len(l_all) == 0:
        return l
    else:
        return [a + b for a, b in zip(l_all, l)]


def multiply_list(l, multiplier):
    return [e * multiplier for e in l]


def histo(l, k=10):
    interval = len(l) * 1.0 / k

    values = []
    for i in range(k):
        start = interval * i
        end = start + interval

        start_int = int(math.ceil(start))
        end_int = int(math.floor(end))

        if end_int >= start_int:
            start_res = start_int - start
            end_res = end - end_int

            mass = sum(l[start_int: end_int])

            if start_res > 0.0000001:
                mass += l[start_int - 1] * start_res

            if end_res > 0.000001:
                mass += l[end_int] * end_res

        else:
            slot = end_int
            portion = end - start

            mass = l[slot] * portion

        values.append(mass / interval)

    return values


def __intruder_auc(org_scores, ext_scores):
    intrusion_label = [1] * len(org_scores) + [0] * len(ext_scores)
    l_score = org_scores + ext_scores

    return roc_auc_score(intrusion_label, l_score)


def test_intruder(mixed_data):
    results = {
        'average_rank': [],
        'average_rank_in_salience': [],
        'average_salience_rank': [],
        'average_salience_rank_in_salience': [],
        'auc': [],
        'salience_auc': [],
        'leading_auc': [],
        'leading_salience_auc': []
    }

    for h_packed_data, l_v_label, org_sizes, ext_sizes in mixed_data:
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
                ext_size = ext_sizes[i]

                l_score = output.data.numpy().tolist()
                l_label = v_label[0].cpu().data.numpy().tolist()

                this_results = analyze_intrusion(l_score, l_label, org_size,
                                                 ext_size)

                for key in this_results:
                    results[key].append(this_results[key])

                ext_labels = l_label[org_size:]

                label_set = set([l for l in ext_labels])

                if len(label_set) == 1:
                    for key in ['auc', 'salience_auc']:
                        results['leading_' + key].append(this_results[key])

    return results


def kernel_word_counting(model, h_packed_data, kernels):
    mtx_e = h_packed_data['mtx_e']
    mtx_evm = h_packed_data['mtx_evm']

    masks = h_packed_data['masks']
    mask_e = masks['mtx_e']
    mask_evm = masks['mtx_evm']

    mtx_e_score = h_packed_data['mtx_e_score']
    mtx_evm_score = h_packed_data['mtx_evm_score']

    ts_e_feature = h_packed_data['ts_e_feature']
    ts_evm_feature = h_packed_data['ts_evm_feature']

    # Combine with node features.
    l_node_features = []
    if ts_e_feature is not None:
        e_node_score = F.tanh(model.node_lr(ts_e_feature))
        l_node_features.append(e_node_score)
    if ts_evm_feature is not None:
        evm_node_score = F.tanh(model.node_lr(ts_evm_feature))
        l_node_features.append(evm_node_score)

    combined_mtx_emb, combined_mtx_emb_mask = model. \
        combined_embedding(mtx_e, mtx_evm, mask_e, mask_evm)
    if model.use_mask:
        masked_mtx_emb = combined_mtx_emb * \
                         combined_mtx_emb_mask.unsqueeze(-1)
    else:
        masked_mtx_emb = combined_mtx_emb

    l_mtx_score = []
    if mtx_e_score is not None:
        l_mtx_score.append(mtx_e_score)

    if mtx_evm_score is not None:
        l_mtx_score.append(mtx_evm_score)

    num_entities = mtx_e.size()[1]

    norm_mtx_emb = nn.functional.normalize(masked_mtx_emb, p=2, dim=-1)
    trans_mtx = torch.matmul(norm_mtx_emb, norm_mtx_emb.transpose(-2, -1))

    trans_data = trans_mtx.cpu().data.squeeze(0).numpy()[num_entities:, :]

    words_around = []
    for k in kernels:
        w_around = zip(*np.nonzero(
            np.logical_and(trans_data > k - 0.1, trans_data < k + 0.1)))
        words_around.append(w_around)

    return words_around


def sort_pair(a, b):
    if a > b:
        return b, a
    else:
        return a, b


def cosine(word1, word2, entity_embedding, event_embedding):
    def emb(word):
        if word.startswith('/m/'):
            return entity_embedding[h_entity_id[word]]
        else:
            return event_embedding[h_event_id[word]]

    embedding1 = emb(word1)
    embedding2 = emb(word2)

    from scipy import spatial
    return 1 - spatial.distance.cosine(embedding1, embedding2)


def check_kernel(test_in_path, out_dir, entity_embedding, event_embedding):
    print('linear weights')
    print(predictor.model.linear.weight)

    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entity_emb = np.load(entity_embedding)
    event_emb = np.load(event_embedding)

    kernels = [-0.3, 0.3, 0.7, 0.9]

    near_counts = [{} for _ in kernels]

    outputs = []
    for kernel in kernels:
        out = open(os.path.join(out_dir, 'kernel_near_%.1f.tsv' % kernel), 'w')
        outputs.append(out)

    with open(test_in_path) as test_in:
        limit = 100
        count = 0
        for line in test_in:
            if predictor.io_parser.is_empty_line(line):
                continue

            h_packed_data, l_v_label = predictor.io_parser.parse_data([line])
            words_around = kernel_word_counting(predictor.model, h_packed_data,
                                                kernels)

            h_info = json.loads(line)
            event_spots = h_info.get('event', {}).get('bodyText', {})
            l_h_ids = event_spots.get('sparse_features', {}
                                      ).get('LexicalHead', [])
            l_h = [h_id_event[id] for id in l_h_ids]

            entity_spots = h_info.get('spot', {}).get('bodyText', {})
            l_e_ids = entity_spots.get('entities', [])
            l_e = [h_id_entity[id] for id in l_e_ids]

            all_items = l_e + l_h

            for w_around, n_count in zip(words_around, near_counts):
                for x, y in w_around:
                    l, r = sort_pair(l_h[x], all_items[y])
                    try:
                        n_count[(l, r)] += 1
                    except KeyError:
                        n_count[(l, r)] = 1

            count += 1

            if count % 1000 == 0:
                logging.info("Processed %d file." % count)

            # if count == limit:
            #     break

    import operator

    # Could check freebase from google trends:
    # https://trends.google.com/trends/explore?q=%2Fm%2F0d0vp3

    for n_count, output in zip(near_counts, outputs):
        sorted_counts = sorted(n_count.items(),
                               key=operator.itemgetter(1), reverse=True)
        print(sorted_counts[:10])
        for (left, right), count in sorted_counts:
            sim = cosine(left, right, entity_emb, event_emb)
            output.write("%s\t%s\t%d\t%.4f\n" % (left, right, count, sim))
        output.close()


if __name__ == '__main__':
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )


    class Main(Configurable):
        test_in = Unicode(help='testing data').tag(config=True)
        model_in = Unicode(help='model to read from').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        study_out = Unicode(help='output dir of the study result').tag(
            config=True)
        num_origins = Int(help='Number origin to study').tag(config=True)
        num_intruders = Int(help='Number intruders for each origin').tag(
            config=True)
        event_id_pickle = Unicode(help='Event id pickle').tag(config=True)
        entity_id_pickle = Unicode(help='Entity id pickle').tag(config=True)
        entity_emb_in = Unicode(help='Entity embedding').tag(config=True)
        event_emb_in = Unicode(help='Event embedding').tag(config=True)


    if 3 != len(sys.argv):
        print(
            "[Usage] [this script] [config] "
            "[study mode(intruder, kernel)]"
        )

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

    if mode == 'intruder':
        logging.info("Mode is %s, running intrusion test", mode)
        intrusion_test(para.test_in, para.study_out,
                       para.num_origins, para.num_intruders)
        # intrusion_test(predictor, para.test_in, para.study_out, 10, 100)
    elif mode == 'kernel':
        logging.info("Mode is %s, running kernel analysis", mode)
        logging.info("Loading event ids.")
        import pickle

        h_event_id = pickle.load(open(para.event_id_pickle))
        h_id_event = dict([(v, k) for k, v in h_event_id.items()])

        h_entity_id = pickle.load(open(para.entity_id_pickle))
        h_id_entity = dict([(v, k) for k, v in h_entity_id.items()])

        check_kernel(para.test_in, para.study_out,
                     para.entity_emb_in, para.event_emb_in)
    else:
        logging.info("Unkown mode %s", mode)
