import gzip
import json
from knowledge4ir.salience.utils.evaluation import SalienceEva


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


def get_predictions(predict_res, content_field, entity_vocab_size, s_e_label,
                    s_evm_label):
    predictions = predict_res['predict'] if 'predict' in predict_res else \
        predict_res[content_field]['predict']
    entities = get_e_labels(predictions, s_e_label, entity_vocab_size)
    events = get_evm_labels(predictions, s_evm_label, entity_vocab_size)
    return entities, events


def load_pairs(docs, f_predict_1, f_predict_2, content_field,
               entity_vocab_size):
    with open_func(docs)(docs) as origin, open_func(f_predict_1)(
            f_predict_1) as pred1, open_func(f_predict_2)(f_predict_2) as pred2:
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

                yield predictions1, predictions2
            except StopIteration:
                break


def compare_ranking(l_e_pack1, l_e_pack2):
    ranks1, wrongs1 = get_rank(l_e_pack1)
    ranks2, wrongs2 = get_rank(l_e_pack2)

    print ranks1
    print ranks2

    print wrongs1
    print wrongs2


def get_rank(l_e_pack):
    num_pos = sum(l_e_pack[1])
    count = 0
    ranks = []
    wrongs = []

    for rank, (score, label, id) in enumerate(zip(*l_e_pack)):
        if label == 1:
            count += 1
            ranks.append((rank, score, id))
        else:
            wrongs.append((rank, score, id))

        if count == num_pos:
            break
    return ranks, wrongs


def compare(docs, f_predict_1, f_predict_2, entity_hash, event_hash,
            entity_vocab_size, content_field='bodyText'):
    print("Comparing predictions [%s] from [%s]." % (f_predict_1, f_predict_2))
    evaluator = SalienceEva()  # evaluator with default values.

    p = 0

    for res in load_pairs(docs, f_predict_1, f_predict_2, content_field,
                          entity_vocab_size):
        p += 1

        (l_e_pack1, l_evm_pack1), (l_e_pack2, l_evm_pack2) = res

        if l_evm_pack1 and l_evm_pack2:
            h_evm1 = evaluator.evaluate(l_evm_pack1[0], l_evm_pack1[1])
            h_evm2 = evaluator.evaluate(l_evm_pack2[0], l_evm_pack2[1])

            compare_ranking(l_evm_pack1, l_evm_pack2)

        if l_e_pack1 and l_e_pack2:
            h_e1 = evaluator.evaluate(l_e_pack1[0], l_e_pack1[1])
            h_e2 = evaluator.evaluate(l_e_pack2[0], l_e_pack2[1])
            compare_ranking(l_e_pack1, l_e_pack2)

        sys.stdin.readline()

        sys.stdout.write('\rCompared %d files' % p)
    print('')


if __name__ == '__main__':
    import sys

    args = sys.argv
    if len(args) < 6:
        print(
            "Usage: [this script] [gold standard] [prediction1] [prediction2] "
            "[entity hash file] [event hash file]"
            "[Default: 723749, entity vocab size]")
        exit(1)

    vocab_size = 723749 if len(args) < 7 else int(args[6])

    compare(args[1], args[2], args[3], args[4], args[5], vocab_size)
