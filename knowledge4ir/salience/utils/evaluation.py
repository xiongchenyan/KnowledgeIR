"""
salience evaluations
"""

from traitlets.config import Configurable
from traitlets import (
    List,
    Int,
    Unicode,
)
import gzip
import json
from knowledge4ir.utils import add_svm_feature, mutiply_svm_feature
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

        for inline, pred_line in zip(origin, pred):
            p += 1

            doc = json.loads(inline)
            l_e = doc['spot']['bodyText']['entities']
            l_label_e = doc['spot']['bodyText']['salience']
            s_e_label = dict(zip(l_e, l_label_e))

            l_evm = doc['event']['bodyText']['sparse_features'].get(
                'LexicalHead', [])
            l_label_evm = doc['event']['bodyText']['salience']
            s_evm_label = dict(zip(l_evm, l_label_evm))

            predict_res = json.loads(pred_line)

            assert doc['docno'] == predict_res['docno']

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
                '\rEvaluated %d files, %d with entities and %d with events' % (
                    p, e_p, evm_p))
        print('')

    h_e_mean_eva = mutiply_svm_feature(h_e_total_eva, 1.0 / e_p)
    logging.info('finished predicted [%d] docs on entity, eva %s', e_p,
                 json.dumps(h_e_mean_eva))

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


class SalienceEva(Configurable):
    l_depth = List(Int, default_value=[1, 5, 10, 20]).tag(config=True)
    l_metrics = List(Unicode, default_value=['p', 'precision', 'recall',
                                             'accuracy']).tag(config=True)

    def __init__(self, **kwargs):
        super(SalienceEva, self).__init__(**kwargs)
        self.h_eva_metric = {
            "p": self.p_at_k,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy
        }

    def evaluate(self, l_score, l_label):
        h_eva_res = {}
        for metric in self.l_metrics:
            h_eva_res.update(self.h_eva_metric[metric](l_score, l_label))
        return h_eva_res

    def p_at_k(self, l_score, l_label):
        h_p = {}
        l_d = zip(l_score, l_label)
        l_d.sort(key=lambda item: -item[0])
        correct = 0

        for p in xrange(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / depth
                h_p['p@%d' % depth] = res
        return h_p

    def precision(self, l_score, l_label):
        z = 0
        c = 0
        for score, label in zip(l_score, l_label):
            if score > 0:
                z += 1
                if label > 0:
                    c += 1
        return {'precision': float(c) / max(z, 1.0)}

    def recall(self, l_score, l_label):
        z = 0
        c = 0
        for score, label in zip(l_score, l_label):
            if label > 0:
                z += 1
                if score > 0:
                    c += 1
        return {'recall': float(c) / max(z, 1.0)}

    def accuracy(self, l_score, l_label):
        c = 0
        for score, label in zip(l_score, l_label):
            if label > 0:
                if score > 0:
                    c += 1
        z = len(l_score)
        return {'accuracy': float(c) / max(z, 1.0)}


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
