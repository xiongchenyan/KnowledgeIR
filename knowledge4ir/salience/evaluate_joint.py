import gzip
import json
from knowledge4ir.utils import add_svm_feature, mutiply_svm_feature
from knowledge4ir.salience.utils.evaluation import SalienceEva
import logging


def open_func(corpus_in):
    return gzip.open if corpus_in.endswith("gz") else open


def evaluate_json_joint(docs, f_predict, entity_vocab_size):
    print("Evaluating joint predictions [%s] from [%s]." % (f_predict, docs))

    evaluator = SalienceEva()  # evaluator with default values.

    h_e_total_eva = dict()
    h_evm_total_eva = dict()

    import sys
    with open_func(docs)(docs) as origin, open_func(f_predict)(
            f_predict) as pred:
        e_p = 0
        evm_p = 0
        p = 0
        skip = 0

        while True:
            try:
                inline = origin.next()
                pred_line = pred.next()

                p += 1
                doc = json.loads(inline)
                predict_res = json.loads(pred_line)

                gold_doc = doc['docno']
                pred_doc = predict_res['docno']

                while not gold_doc == pred_doc:
                    # Some results may have skipped empty lines.
                    skip += 1
                    doc = json.loads(origin.next())
                    gold_doc = doc['docno']

                l_e = doc['spot']['bodyText']['entities']
                l_label_e = doc['spot']['bodyText']['salience']
                s_e_label = dict(zip(l_e, l_label_e))

                l_evm = doc['event']['bodyText']['sparse_features'].get(
                    'LexicalHead', [])
                l_label_evm = doc['event']['bodyText']['salience']
                s_evm_label = dict(zip(l_evm, l_label_evm))

                l_e_pack, l_evm_pack = split_joint_list(entity_vocab_size,
                                                        predict_res['predict'],
                                                        s_e_label, s_evm_label)

                if l_e_pack:
                    h_e = evaluator.evaluate(l_e_pack[0], l_e_pack[1])
                    e_p += 1
                    h_e_total_eva = add_svm_feature(h_e_total_eva, h_e)

                if l_evm_pack:
                    h_evm = evaluator.evaluate(l_evm_pack[0], l_evm_pack[1])
                    evm_p += 1
                    h_evm_total_eva = add_svm_feature(h_evm_total_eva, h_evm)

                sys.stdout.write(
                    '\rEvaluated %d files, %d with entities and %d with events,'
                    ' %d line skipped.' % (p, e_p, evm_p, skip))

            except StopIteration:
                break
        print('')

    h_e_mean_eva = {}
    if not e_p == 0:
        h_e_mean_eva = mutiply_svm_feature(h_e_total_eva, 1.0 / e_p)
        logging.info('finished predicted [%d] docs on entity, eva %s', e_p,
                     json.dumps(h_e_mean_eva))

    h_evm_mean_eva = {}
    if not evm_p == 0:
        h_evm_mean_eva = mutiply_svm_feature(h_evm_total_eva, 1.0 / evm_p)
        logging.info('finished predicted [%d] docs on event, eva %s', evm_p,
                     json.dumps(h_evm_mean_eva))

    res = [h_e_mean_eva, h_evm_mean_eva]

    with open(f_predict + '.joint.eval', 'w') as out:
        json.dump(res, out, indent=1)


def split_joint_list(entity_vocab_size, predictions, s_e_label, s_evm_label):
    e_list = []
    evm_list = []

    for pred in predictions:
        eid = pred[0]
        score = pred[1][0]
        if eid >= entity_vocab_size:
            evm_list.append((score, s_evm_label[eid - entity_vocab_size],
                             eid - entity_vocab_size))
        else:
            e_list.append((score, s_e_label[eid], eid))

    return zip(*e_list), zip(*evm_list)


if __name__ == '__main__':
    import sys

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    args = sys.argv
    if not len(args) == 4:
        print(
            "Usage: [this script] [gold standard] [prediction]"
            "[entity vocab size]")
        exit(1)

    evaluate_json_joint(args[1], args[2], int(args[3]))
