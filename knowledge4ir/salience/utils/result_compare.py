from __future__ import print_function
import gzip
import json
from knowledge4ir.salience.utils.evaluation import SalienceEva
import logging
import pickle


def open_func(corpus_in):
    return gzip.open if corpus_in.endswith("gz") else open


def get_e_labels(predictions, s_e_label, entity_vocab_size):
    e_list = []

    for pred in predictions:
        eid = pred[0]
        score = pred[1]
        if eid < entity_vocab_size:
            e_list.append((score, s_e_label[eid], eid))

    e_list.sort(reverse=True)
    return zip(*e_list)


def get_evm_labels(predictions, s_evm_label, entity_vocab_size):
    evm_list = []

    for pred in predictions:
        eid = pred[0]
        score = pred[1]
        if eid >= entity_vocab_size:
            evm_list.append((score, s_evm_label[eid - entity_vocab_size],
                             eid - entity_vocab_size))
    evm_list.sort(reverse=True)
    return zip(*evm_list)


def get_predictions(predict_res, content_field, entity_vocab_size,
                    s_e_label,
                    s_evm_label):
    predictions = predict_res['predict'] if 'predict' in predict_res else \
        predict_res[content_field]['predict']
    entities = get_e_labels(predictions, s_e_label, entity_vocab_size)
    events = get_evm_labels(predictions, s_evm_label, entity_vocab_size)
    return entities, events


def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])


class ResultComparer():
    def __init__(self, word_id_pickle_in, entity_id_pickle_in,
                 event_id_pickle_in):
        self.h_word = invert_dict(pickle.load(open(word_id_pickle_in)))
        self.h_entity = invert_dict(pickle.load(open(entity_id_pickle_in)))
        self.h_event = invert_dict(pickle.load(open(event_id_pickle_in)))

        print('loaded [%d] word ids, [%d] entity ids, [%d] event ids.' % (
            len(self.h_word), len(self.h_entity), len(self.h_event)))

    def load_pairs(self, docs, f_predict_1, f_predict_2, content_field,
                   entity_vocab_size):
        with open_func(docs)(docs) as origin, \
                open_func(f_predict_1)(f_predict_1) as pred1, \
                open_func(f_predict_2)(f_predict_2) as pred2:
            while True:
                try:
                    inline = origin.next()
                    pred_line1 = pred1.next()
                    pred_line2 = pred2.next()

                    doc = json.loads(inline)
                    predict_res1 = json.loads(pred_line1)
                    predict_res2 = json.loads(pred_line2)

                    gold_doc = doc['docno']
                    pred_doc1 = predict_res1['docno']
                    pred_doc2 = predict_res2['docno']

                    if not (gold_doc == pred_doc1 and gold_doc == pred_doc2):
                        # Do not handle inconsistent lines.
                        raise StopIteration

                    l_e = doc['spot']['bodyText']['entities']
                    l_label_e = doc['spot']['bodyText']['salience']
                    s_e_label = dict(zip(l_e, l_label_e))

                    l_evm = doc['event']['bodyText']['sparse_features'].get(
                        'LexicalHead', [])
                    l_label_evm = doc['event']['bodyText']['salience']
                    s_evm_label = dict(zip(l_evm, l_label_evm))

                    predictions1 = get_predictions(predict_res1, content_field,
                                                   entity_vocab_size, s_e_label,
                                                   s_evm_label)
                    predictions2 = get_predictions(predict_res2, content_field,
                                                   entity_vocab_size, s_e_label,
                                                   s_evm_label)

                    yield doc, predictions1, predictions2
                except StopIteration:
                    break

    def reveal(self, r_list, d):
        return [(rank, score, d[id]) for (rank, score, id) in r_list]

    def get_rank(self, l_e_pack):
        num_pos = sum(l_e_pack[1])
        count = 0
        ranks = []
        wrongs = []

        for rank, (score, label, eid) in enumerate(zip(*l_e_pack)):
            if label == 1:
                count += 1
                ranks.append((rank, score, eid))
            else:
                wrongs.append((rank, score, eid))

            if count == num_pos:
                break
        return ranks, wrongs

    def compare_ranking(self, l_e_pack1, l_e_pack2, e_ids):
        ranks1, wrongs1 = self.get_rank(l_e_pack1)
        ranks2, wrongs2 = self.get_rank(l_e_pack2)

        print("Positive ranking positions.")
        print(self.reveal(ranks1, e_ids))
        print(self.reveal(ranks2, e_ids))

        print("Errors high in rank list.")
        print(self.reveal(wrongs1, e_ids))
        print(self.reveal(wrongs2, e_ids))

    def show_graph(self, l_e_pack, l_evm_pack, adjacent):
        full_adj = zip(zip(*l_evm_pack), adjacent)
        pos_entities = set([eid for score, label, eid in zip(*l_e_pack) if
                            label == 1])
        pos_event_adj = []
        pos_entity_adj = []
        for (score, label, evm), adj in full_adj:
            saliences = [(self.h_entity[e], e in pos_entities) for e in adj]
            if any(pos for e, pos in saliences):
                pos_entity_adj.append((self.h_event[evm], label, saliences))
            if label == 1:
                pos_event_adj.append((self.h_event[evm], label, saliences))

        return pos_event_adj, pos_entity_adj

    def get_targets(self, doc):
        words = [self.h_word[w] for w in doc['bodyText']]
        entities = [self.h_entity[e] for e in
                    doc['spot']['bodyText']['entities']]
        events = [self.h_event[e] for e in
                  doc['event']['bodyText']['sparse_features']['LexicalHead']]
        adjacent = [l_adj for l_adj in doc['adjacent']]
        return words, entities, events, adjacent

    def compare(self, docs, f_predict_1, f_predict_2,
                entity_vocab_size, content_field='bodyText'):
        print("Comparing predictions [%s] from [%s]." % (
            f_predict_1, f_predict_2))
        evaluator = SalienceEva()  # evaluator with default values.

        p = 0

        for res in self.load_pairs(docs, f_predict_1, f_predict_2,
                                   content_field, entity_vocab_size):
            p += 1

            doc, (l_e_pack1, l_evm_pack1), (l_e_pack2, l_evm_pack2) = res
            words, entities, events, adjacent = self.get_targets(doc)

            print('Comparing doc %s' % (doc['docno']))

            print('Words are:')
            print(' '.join(words))
            print('Events are:')
            print(', '.join(events))

            if l_evm_pack1 and l_evm_pack2:
                print('comparing event ranking.')
                # h_evm1 = evaluator.evaluate(l_evm_pack1[0], l_evm_pack1[1])
                # h_evm2 = evaluator.evaluate(l_evm_pack2[0], l_evm_pack2[1])
                self.compare_ranking(l_evm_pack1, l_evm_pack2, self.h_event)
                print('showing adjacent list.')
                print([item for item in zip(events, adjacent) if item[1]])

            if l_e_pack1 and l_e_pack2:
                print('comparing entity ranking.')
                # h_e1 = evaluator.evaluate(l_e_pack1[0], l_e_pack1[1])
                # h_e2 = evaluator.evaluate(l_e_pack2[0], l_e_pack2[1])
                self.compare_ranking(l_e_pack1, l_e_pack2, self.h_entity)

            if l_e_pack1 and l_evm_pack1:
                print('Showing graph.')
                evm_adj, e_adj = self.show_graph(l_e_pack1, l_evm_pack1,
                                                 adjacent)
                print('Salient event adjacent:')
                print(evm_adj)
                print('Salient entity adjacent:')
                print(e_adj)

            sys.stdin.readline()

        #     sys.stdout.write('\rCompared %d files' % p)
        # print('')


if __name__ == '__main__':
    import sys

    args = sys.argv
    if len(args) < 6:
        print(
            "Usage: [this script] [gold standard] [prediction1] [prediction2] "
            "[word hash file] [entity hash file] [event hash file]"
            "[Default: 723749, entity vocab size]")
        exit(1)

    vocab_size = 723749 if len(args) < 8 else int(args[7])

    gold, pred1, pred2, word_hash, entity_hash, event_hash = args[1:7]

    comparer = ResultComparer(word_hash, entity_hash, event_hash)
    comparer.compare(gold, pred1, pred2, vocab_size)
