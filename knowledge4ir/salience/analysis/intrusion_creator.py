from __future__ import print_function
import json
from collections import Counter
import sys
import itertools
from collections import defaultdict


def apply_selector(events, selections):
    new_events = {
        'sparse_features': {
            'LexicalHead': [],
            'SparseFrameName': [],
        },
        'salience': [0] * len(selections),
        'features': [],
    }

    for index in selections:
        new_events['sparse_features']['LexicalHead'].append(
            events['sparse_features']['LexicalHead'][index]
        )
        new_events['sparse_features']['SparseFrameName'].append(
            events['sparse_features']['SparseFrameName'][index]
        )
        new_events['features'].append(
            events['features'][index]
        )
    return new_events


def increment_merge(origin_events, intruder_events, adjuster):
    l_merged_events = []
    selected_until = []

    for until in range(1, len(intruder_events['salience']) + 1):
        merged_events = {
            'sparse_features': {
                'LexicalHead': [],
                'SparseFrameName': [],
            },
            'salience': [1] * len(origin_events['salience']) + [0] * until,
            'features': [],
            'origin_salience': origin_events['salience']
        }

        merged_events['sparse_features']['LexicalHead'].extend(
            origin_events['sparse_features']['LexicalHead']
        )
        merged_events['sparse_features']['SparseFrameName'].extend(
            origin_events['sparse_features']['SparseFrameName']
        )
        merged_events['features'].extend(
            origin_events['features']
        )

        merged_events['sparse_features']['LexicalHead'].extend(
            intruder_events['sparse_features'][
                'LexicalHead'][:until]
        )
        merged_events['sparse_features']['SparseFrameName'].extend(
            intruder_events['sparse_features'][
                'SparseFrameName'][:until]
        )

        intruder_features = intruder_events['features'][:until]

        adjusted_features = []

        for index, features in enumerate(intruder_features):
            freq = features[-2] + adjuster[index]
            new_f = [0] * len(features)
            new_f[-2] = freq
            adjusted_features.append(new_f)

        merged_events['features'].extend(
            adjusted_features
        )

        l_merged_events.append(merged_events)
        selected_until.append(until)

    return l_merged_events, selected_until


def merge_entities(origin_doc, intruding_doc, events, selector=None):
    new_doc = {
        'origin': origin_doc['docno'],
        'intruder': origin_doc['docno'],
        'bodyText': [],
        'abstract': [],
        'spot': {
            'bodyText': {
                'entities': [],
                'salience': {},
                'features': {}
            },
        },
        'event': {'bodyText': events, }
    }

    for key in new_doc['spot']['bodyText']:
        new_doc['spot']['bodyText'][
            key] = origin_doc['spot']['bodyText'][key][:]

    adjacent_map = intruding_doc['adjacent']

    if selector:
        eid2indices = defaultdict(list)
        intruding_entities = intruding_doc['spot']['bodyText']['entities']
        for index, eid in enumerate(intruding_entities):
            eid2indices[eid].append(index)

        selected_entity_indices = set()
        adjacent = []
        for index in selector:
            adjacent.append(adjacent_map[index])
            for eid in adjacent_map[index]:
                selected_entity_indices.update(eid2indices[eid])

        sorted_indices = sorted(selected_entity_indices)

        new_doc['adjacent'] = adjacent

        for key in new_doc['spot']['bodyText']:
            for index in sorted_indices:
                new_doc['spot']['bodyText'][key].append(
                    intruding_doc['spot']['bodyText'][key][index])
    else:
        new_doc['adjacent'] = origin_doc['adjacent']

    return new_doc


def mix(origin_doc, intruding_doc):
    intruder_events = intruding_doc['event']['bodyText']
    intruder_heads = intruder_events['sparse_features']['LexicalHead']

    head_count = Counter()
    count_till_here = []
    for head in intruder_heads:
        head_count[head] += 1
        count_till_here.append(head_count[head])

    origin_events = origin_doc['event']['bodyText']
    origin_freq = [f[-2] for f in origin_events['features']]
    origin_heads = origin_events['sparse_features']['LexicalHead']

    origin_hc = dict(zip(origin_heads, origin_freq))

    intruder_hc = [head_count[head] for head in intruder_heads]
    intruder_origin_hc = [origin_hc.get(head, 0) for head in intruder_heads]

    freq_adjuster = [fc - fi + fo for (fi, fo, fc) in
                     zip(intruder_hc, intruder_origin_hc, count_till_here)]

    intruder_saliency = intruder_events['salience']

    salient_indices = []
    sa_adjuster = []

    non_salient_indices = []
    non_sa_adjuster = []

    for index, s in enumerate(intruder_saliency):
        if s == 1:
            salient_indices.append(index)
            sa_adjuster.append(freq_adjuster[index])
        else:
            non_salient_indices.append(index)
            non_sa_adjuster.append(freq_adjuster[index])

    sa_events = apply_selector(intruder_events, salient_indices)
    non_sa_events = apply_selector(intruder_events, non_salient_indices)

    origin_events = origin_doc['event']['bodyText']
    sa_set, sa_until = increment_merge(origin_events, sa_events, sa_adjuster)
    non_sa_set, non_sa_until = increment_merge(origin_events, non_sa_events,
                                               non_sa_adjuster)

    full_sa_set = []
    for events, until in zip(sa_set, sa_until):
        full_sa_set.append(
            merge_entities(origin_doc, intruding_doc, events,
                           salient_indices[:until])
        )

    full_non_sa_set = []
    for events, until in zip(non_sa_set, non_sa_until):
        full_non_sa_set.append(
            merge_entities(origin_doc, intruding_doc, events,
                           non_salient_indices[:until])
        )

    return full_sa_set, full_non_sa_set


def create_intruders(hashed_corpus, output_path, num_origin, num_intruder_per):
    count = 0
    intruder_count = 0

    origins = []

    with open(hashed_corpus) as data_in, open(output_path, 'w') as data_out:
        for line in data_in:
            doc_info = json.loads(line)
            salience_labels = doc_info['event']['bodyText']['salience']

            num_salience = sum(salience_labels)

            if num_salience < 5:
                # Skip documents less than 5 salient ones.
                continue

            if len(salience_labels) > 100:
                # Skip documents with more than 100 events
                # (to avoid large mixed doc).
                continue

            if count < num_origin:
                origins.append(doc_info)
            elif count >= num_origin and intruder_count < num_intruder_per:
                intruder_count += 1
                for i, origin in enumerate(origins):
                    sa_docs, non_sa_docs = mix(origin, doc_info)
                    test_pack = {
                        'salient_test': sa_docs,
                        'non_salient_test': non_sa_docs,
                    }

                    print(
                        "\rProcessed %d intruders for origin %d" % (
                            intruder_count, i + 1
                        ), end=''
                    )
                    data_out.write(json.dumps(test_pack) + '\n')

            else:
                print('\nDone.')
                break

            count += 1


if __name__ == '__main__':
    corpus_path, output_path, num_origin, num_intruder = sys.argv[1:5]
    num_origin = int(num_origin)
    num_intruder = int(num_intruder)

    create_intruders(corpus_path, output_path, num_origin, num_intruder)
