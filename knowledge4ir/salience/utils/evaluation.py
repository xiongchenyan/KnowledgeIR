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
from sklearn.metrics import roc_auc_score
import numpy as np


class SalienceEva(Configurable):
    l_depth = List(Int, default_value=[1, 5, 10, 20]).tag(config=True)
    l_metrics = List(Unicode, default_value=['p', 'r', 'auc']).tag(config=True)

    def __init__(self, **kwargs):
        super(SalienceEva, self).__init__(**kwargs)
        self.h_eva_metric = {
            "p": self.p_at_k,
            'r': self.r_at_k,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "auc": self.auc,
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
                h_p['p@%02d' % depth] = res
        return h_p

    def r_at_k(self, l_score, l_label):
        h_r = {}
        l_d = zip(l_score, l_label)
        l_d.sort(key=lambda item: -item[0])
        correct = 0
        total_z = max(1, sum([max(0, min(label, 1)) for label in l_label]))
        for p in xrange(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / total_z
                h_r['r@%02d' % depth] = res
        return h_r

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

    def auc(self, l_score, l_label):
        l_label = [max(0, item) for item in l_label]  # binary
        l_label = [min(1, item) for item in l_label]
        if min(l_label) == 1:
            auc_score = 1
        elif max(l_label) == 0:
            auc_score = 0
        else:
            auc_score = roc_auc_score(l_label, l_score)
        return {'auc': auc_score}
