from __future__ import print_function
import gzip
import json
from knowledge4ir.salience.utils.evaluation import SalienceEva
import logging
import pickle


def open_func(corpus_in):
    return gzip.open if corpus_in.endswith("gz") else open


class GraphChecker():
    def __init__(self):
        self.num_pos_with_s_args = 0
        self.num_pos_without_s_args = 0
        self.num_neg_with_s_args = 0
        self.num_neg_without_s_args = 0
        self.num_events = 0

    def check_graph(self, l_e_pack, l_evm_pack, adjacent):
        pos_entities = set([eid for eid, label in l_e_pack if label == 1])

        for (evm, label), adj in zip(l_evm_pack, adjacent):
            if any([e in pos_entities for e in adj]):
                if label == 1:
                    self.num_pos_with_s_args += 1
                else:
                    self.num_neg_with_s_args += 1
            else:
                if label == 1:
                    self.num_pos_without_s_args += 1
                else:
                    self.num_pos_without_s_args += 1
            self.num_events += 1

    def check(self, docs):
        p = 0

        with open_func(docs)(docs) as origin:
            for inline in origin:
                doc = json.loads(inline)
                l_e = doc['spot']['bodyText']['entities']
                l_label_e = doc['spot']['bodyText']['salience']
                l_e_labels = zip(l_e, l_label_e)

                l_evm = doc['event']['bodyText']['sparse_features'].get(
                    'LexicalHead', [])
                l_label_evm = doc['event']['bodyText']['salience']
                l_evm_labels = zip(l_evm, l_label_evm)

                if not l_evm_labels:
                    continue

                p += 1
                self.num_events += len(l_evm_labels)

                adjacent = [l_adj for l_adj in doc['adjacent']]

                self.check_graph(l_e_labels, l_evm_labels, adjacent)

                prec = 1.0 * self.num_pos_with_s_args / (
                        self.num_pos_without_s_args + self.num_neg_with_s_args)

                recall = 1.0 * self.num_pos_with_s_args / (
                        self.num_pos_with_s_args + self.num_pos_without_s_args)

                sys.stdout.write(
                    "\rProcessed %d documents, positives:%d, "
                    "all events: %d, prec: %.4f, recall: %.4f" % (
                        p, self.num_pos_with_s_args, self.num_events, prec,
                        recall))

        print("\nResults:")
        print("\tWith entities\tWithout entities")
        print("Positives\t%d\t%d" % (
            self.num_pos_with_s_args, self.num_pos_without_s_args))
        print("Negatives\t%d\t%d" % (
            self.num_neg_with_s_args, self.num_neg_without_s_args))


if __name__ == '__main__':
    import sys

    args = sys.argv
    if len(args) < 2:
        print("Usage: [this script] [gold standard with graph]")
        exit(1)

    vocab_size = 723749 if len(args) < 3 else int(args[2])

    gold = args[1]

    comparer = GraphChecker()
    comparer.check(gold)
