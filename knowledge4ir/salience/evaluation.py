"""
salience evaluations
"""

from traitlets.config import Configurable
from traitlets import (
    List,
    Int,
    Unicode,
)


class SalienceEva(Configurable):
    l_depth = List(Int, default_value=[1, 5, 10, 20]).tag(config=True)
    l_metrics = List(Unicode, default_value=['p']).tag(config=True)

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
            if p in self.l_depth:
                res = float(correct) / p
                h_p['p@%d' % p] = res
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
        return {'recall': float(c) / max(z, 1.0)}




