"""
data io
"""
import json

import torch
from torch.autograd import Variable

from knowledge4ir.salience.center import use_cuda
from knowledge4ir.utils import term2lm


def padding(ll, filler):
    n = max([len(l) for l in ll])
    for i in xrange(len(ll)):
        ll[i] += [filler] * (n - len(ll[i]))
    return ll


def get_top_k_e(l_e, max_e_per_d):
    h_e_tf = term2lm(l_e)
    l_e_tf = sorted(h_e_tf.items(), key=lambda item: -item[1])[:max_e_per_d]
    l_e = [item[0] for item in l_e_tf]
    z = float(sum([item[1] for item in l_e_tf]))
    l_w = [item[1] / z for item in l_e_tf]
    return l_e, l_w


def raw_io(l_line, spot_field, in_field, salience_field, max_e_per_d=200):
    """
    convert data to the input for the model
    """
    ll_e = []
    ll_w = []
    ll_label = []
    for line in l_line:
        h = json.loads(line)
        l_e = h[spot_field].get(in_field, [])
        l_e, l_w = get_top_k_e(l_e, max_e_per_d)
        s_salient_e = set(h[spot_field].get(salience_field, []))
        l_label = [1 if e in s_salient_e else -1 for e in l_e]
        ll_e.append(l_e)
        ll_w.append(l_w)
        ll_label.append(l_label)

    ll_e = padding(ll_e, 0)
    ll_w = padding(ll_w, 0)
    ll_label = padding(ll_label, 0)
    m_e = Variable(torch.LongTensor(ll_e)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_e))
    m_w = Variable(torch.FloatTensor(ll_w)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_w))
    m_label = Variable(torch.FloatTensor(ll_label)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_label))

    h_packed_data = {
        "mtx_e": m_e,
        "mtx_score": m_w
    }
    return h_packed_data, m_label


def feature_io(l_line, spot_field, in_field, salience_field, max_e_per_d=None):
    """
    io with pre-filtered entity list and feature matrices
    """
    ll_e = []
    lll_feature = []
    ll_label = []
    f_dim = 0
    for line in l_line:
        h = json.loads(line)
        packed = h[spot_field].get(in_field, {})
        l_e = packed.get('entities', [])
        ll_feature = packed.get('features', [])
        if not l_e:
            continue
        if ll_feature:
            f_dim = max(f_dim, len(ll_feature[0]))
        s_salient_e = set(h[spot_field].get(salience_field, {}).get('entities', []))
        l_label = [1 if e in s_salient_e else -1 for e in l_e]
        ll_e.append(l_e)
        ll_label.append(l_label)
        lll_feature.append(ll_feature)

    ll_e = padding(ll_e, 0)
    ll_label = padding(ll_label, 0)
    lll_feature = padding(lll_feature, [0] * f_dim)
    m_e = Variable(torch.LongTensor(ll_e)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_e))
    m_label = Variable(torch.FloatTensor(ll_label)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_label))
    ts_feature = Variable(torch.FloatTensor(lll_feature)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(lll_feature))

    h_packed_data = {
        "mtx_e": m_e,
        "ts_feature": ts_feature
    }

    return h_packed_data, m_label