import json
from collections import Counter


def apply_selector(events, selections):
    new_events = {
        'bodyText': {
            'sparse_features': {
                'LexicalHead': [],
                'SparseFrameName': [],
            },
            'salience': [0] * len(selections),
            'features': [],
        }
    }

    for index in selections:
        new_events['bodyText']['sparse_features']['LexicalHead'].append(
            events['bodyText']['sparse_features']['LexicalHead'][index]
        )
        new_events['bodyText']['sparse_features']['SparseFrameName'].append(
            events['bodyText']['sparse_features']['SparseFrameName'][index]
        )
        new_events['bodyText']['features'].append(
            events['bodyText']['features'][index]
        )

    return events


def increment_merge(origin_events, intruder_events, adjuster):
    l_merged_events = []

    num_intruder = len(intruder_events['salience'])

    for until in range(1, num_intruder + 1):
        merged_events = {
            'bodyText': {
                'sparse_features': {
                    'LexicalHead': [],
                    'SparseFrameName': [],
                },
                'salience': [1] * len(origin_events['salience']) + [0] * until,
                'features': [],
            }
        }

        merged_events['bodyText']['sparse_features']['LexicalHead'].extend(
            origin_events['bodyText']['sparse_features']['LexicalHead']
        )
        merged_events['bodyText']['sparse_features']['SparseFrameName'].extend(
            origin_events['bodyText']['sparse_features']['SparseFrameName']
        )
        merged_events['bodyText']['features'].extend(
            origin_events['bodyText']['features']
        )

        merged_events['bodyText']['sparse_features']['LexicalHead'].extend(
            intruder_events['bodyText']['sparse_features'][
                'LexicalHead'][:until]
        )
        merged_events['bodyText']['sparse_features']['SparseFrameName'].extend(
            intruder_events['bodyText']['sparse_features'][
                'SparseFrameName'][:until]
        )

        intruder_features = intruder_events['bodyText']['features'][:until]

        adjusted_features = []

        for index, features in enumerate(intruder_features):
            new_f = features[:]
            new_f[-2] -= adjuster[index]
            adjusted_features.append(new_f)

        merged_events['bodyText']['features'].extend(
            adjusted_features
        )

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

    origin_events = origin_doc['event']['docText']
    sa_set = increment_merge(origin_events, sa_events, sa_adjuster)
    non_sa_set = increment_merge(origin_events, non_sa_events, non_sa_adjuster)

    return add_meta(origin_doc, sa_set), add_meta(origin_doc, non_sa_set)


def create_intruders(hashed_corpus, output_path, num_origin, num_interder_per):
    count = 0

    origins = []
    intruders = []

    num_intruders = num_interder_per * num_origin

    with open(hashed_corpus) as data_in, open(output_path, 'w') as data_out:
        for line in data_in:
            doc_info = json.load(line)
            salience_labels = doc_info['events']['bodyText']['salience']

            num_salience = sum(salience_labels)

            if num_salience < 5:
                continue

            if count < num_origin:
                origins.append(doc_info)
            elif count < num_intruders + num_origin:
                for origin in origins:
                    sa_docs, non_sa_docs = mix(origin, doc_info)
            else:
                break

            count += 1
