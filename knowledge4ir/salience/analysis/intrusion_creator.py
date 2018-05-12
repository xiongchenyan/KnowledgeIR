from __future__ import print_function
import json
from collections import Counter
import sys


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

    for until in range(1, len(intruder_events['salience']) + 1):
        merged_events = {
            'sparse_features': {
                'LexicalHead': [],
                'SparseFrameName': [],
            },
            'salience': [1] * len(origin_events['salience']) + [0] * until,
            'features': [],
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
            freq = features[-2] - adjuster[index]
            new_f = [0] * len(features)
            new_f[-2] = freq
            adjusted_features.append(new_f)

        merged_events['features'].extend(
            adjusted_features
        )

        l_merged_events.append(merged_events)

    return l_merged_events


def add_meta(origin_doc, events, selector=None):
    new_doc = {'bodyText': [], 'abstract': [], 'spot': [], 'event': events}

    if selector:
        adjacent = [origin_doc['adjacent'][index] for index in selector]
        new_doc['adjacent'] = adjacent
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

    indiced_count = [head_count[head] for head in intruder_heads]
    count_adjuster = [x - y for (x, y) in zip(indiced_count, count_till_here)]

    intruder_saliency = intruder_events['salience']

    salient_indices = []
    sa_adjuster = []

    non_salient_indices = []
    non_sa_adjuster = []

    for index, s in enumerate(intruder_saliency):
        if s == 1:
            salient_indices.append(index)
            sa_adjuster.append(count_adjuster[index])
        else:
            non_salient_indices.append(index)
            non_sa_adjuster.append(count_adjuster[index])

    sa_events = apply_selector(intruder_events, salient_indices)

    non_sa_events = apply_selector(intruder_events, non_salient_indices)

    origin_events = origin_doc['event']['bodyText']
    sa_set = increment_merge(origin_events, sa_events, sa_adjuster)
    non_sa_set = increment_merge(origin_events, non_sa_events, non_sa_adjuster)

    meta_sa_set = [add_meta(origin_doc, doc) for doc in sa_set]
    meta_non_sa_set = [add_meta(origin_doc, doc) for doc in non_sa_set]

    return meta_sa_set, meta_non_sa_set


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
