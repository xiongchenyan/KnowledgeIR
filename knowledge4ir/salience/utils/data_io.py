"""
data io
"""
import json

import torch
from torch.autograd import Variable
from knowledge4ir.utils import (
    term2lm,
    SPOT_FIELD,
    EVENT_SPOT_FIELD,
    body_field,
    title_field,
    abstract_field,
    salience_gold,
    adjacent_field
)
import numpy as np
import logging
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode,
    List,
)

use_cuda = torch.cuda.is_available()


class DataIO(Configurable):
    nb_features = Int(help='number of features to use').tag(config=True)
    spot_field = Unicode(SPOT_FIELD, help='spot field').tag(config=True)
    event_spot_field = Unicode(EVENT_SPOT_FIELD, help='event spot field').tag(
        config=True)
    salience_label_field = Unicode(salience_gold, help='salience label').tag(
        config=True)
    salience_field = Unicode(abstract_field, help='salience field').tag(
        config=True)
    adjacent_field = Unicode(adjacent_field, help='adjacent field').tag(
        config=True)
    max_e_per_d = Int(200, help='max entity per doc').tag(config=True)
    content_field = Unicode(body_field, help='content field').tag(config=True)
    max_w_per_d = Int(500, help='maximum words per doc').tag(config=True)
    l_target_data = List(
        Unicode,
        default_value=[]
    ).tag(config=True)
    group_name = Unicode(help='hot key for l_target_data').tag(config=True)
    entity_vocab_size = Int(help='vocabulary size of entity').tag(config=True)
    e_feature_dim = Int(help='entity feature dimension').tag(config=True)
    evm_feature_dim = Int(help='event feature dimension').tag(config=True)

    def __init__(self, **kwargs):
        super(DataIO, self).__init__(**kwargs)

        self.h_target_group = {
            'raw': ['mtx_e', 'mtx_score', 'label'],
            'feature': ['mtx_e', 'mtx_score', 'ts_feature', 'label'],
            'duet': ['mtx_e', 'mtx_score', 'mtx_w', 'mtx_w_score', 'label'],
            'joint_raw': ['mtx_e', 'mtx_score', 'label'],
        }

        self.h_data_meta = {
            'mtx_e': {'dim': 2, 'd_type': 'Int'},
            'mtx_score': {'dim': 2, 'd_type': 'Float'},
            'label': {'dim': 2, 'd_type': 'Float'},
            'mtx_w': {'dim': 2, 'd_type': 'Int'},
            'mtx_w_score': {'dim': 2, 'd_type': 'Float'},
            'ts_feature': {'dim': 3, 'd_type': 'Float'},
        }

        self._data_config()

        if not self.l_target_data:
            if self.group_name:
                self.config_target_group()

    def _data_config(self):
        pass

    def config_target_group(self):
        logging.info('io configing via group [%s]', self.group_name)
        self.l_target_data = self.h_target_group[self.group_name]
        print self.h_target_group[self.group_name]
        logging.info('io targets %s', json.dumps(self.l_target_data))

    def is_empty_line(self, line):
        h = json.loads(line)
        l_e = h[self.spot_field].get(self.content_field)
        if type(l_e) == dict:
            l_e = l_e.get('entities')
        return not l_e

    def parse_data(self, l_line):
        l_data = []
        while len(l_data) < len(self.l_target_data):
            l_data.append([])
        h_parsed_data = dict(zip(self.l_target_data, l_data))
        for line in l_line:
            h_info = json.loads(line)
            h_this_data = self._parse_entity(h_info)

            if 'mtx_w' in h_parsed_data:
                h_this_data.update(self._parse_word(h_info))

            for key in h_parsed_data.keys():
                assert key in h_this_data
                h_parsed_data[key].append(h_this_data[key])

        h_parsed_data = self._canonicalize_data(h_parsed_data)
        return h_parsed_data, h_parsed_data['label']

    def _canonicalize_data(self, h_parsed_data):
        for key in h_parsed_data:
            # logging.debug('line [%s]', l_line[0])
            # logging.info('converting [%s] to torch variable', key)
            dim = self.h_data_meta[key]['dim']
            if not self._is_empty(h_parsed_data[key], dim):
                h_parsed_data[key] = self._data_to_variable(
                    self._padding(h_parsed_data[key], dim),
                    data_type=self.h_data_meta[key]['d_type']
                )
            else:
                # Cannot convert empty lists to variable.
                h_parsed_data[key] = None

        return h_parsed_data

    def _data_to_variable(self, list_data, data_type='Float'):
        v = None
        if data_type == 'Int':
            v = Variable(torch.LongTensor(list_data))
        elif data_type == 'Float':
            v = Variable(torch.FloatTensor(list_data))
        else:
            logging.error(
                'convert to variable with data_type [%s] not implemented',
                data_type)
            raise NotImplementedError
        if use_cuda:
            v = v.cuda()
        return v

    def _parse_entity(self, h_info):
        entity_spots = h_info.get(self.spot_field, {}).get(self.content_field,
                                                           {})
        if type(entity_spots) is list:
            # backward compatibility
            return self._parse_entity_old(h_info)

        l_e = entity_spots.get('entities', [])
        test_label = entity_spots[self.salience_label_field]
        l_label_org = [1 if label > 0 else -1 for label in test_label]
        # Associate label with eid.
        h_labels = dict(zip(l_e, l_label_org))
        ll_feature = entity_spots.get('features', [[]] * len(l_e))
        h_e_feature = dict(zip(l_e, ll_feature))
        l_e_tf = [item[0] for item in ll_feature]

        l_e_with_tf = zip(l_e, l_e_tf)
        l_e_with_tf.sort(key=lambda item: item[1], reverse=True)
        l_e_with_tf = l_e_with_tf[:self.max_e_per_d]
        l_kept_e = [item[0] for item in l_e_with_tf]
        l_kept_e_tf = [item[1] for item in l_e_with_tf]

        ll_kept_feature = [h_e_feature[e] for e in l_kept_e]
        l_label = [h_labels[e] for e in l_kept_e]

        if len(l_label) == 0:
            # Early version of Pytorch does not support empty tensor. We add a
            # dummy entity if the document is empty.
            l_label = [0]
            l_kept_e_tf = [0]
            ll_kept_feature = [[0] * self.e_feature_dim]
            l_kept_e = [0]

        h_res = {
            'mtx_e': l_kept_e,
            'mtx_score': l_kept_e_tf,
            'ts_feature': ll_kept_feature,
            'label': l_label
        }
        return h_res

    def _parse_entity_old(self, h_info):
        # backward compatibility
        entity_spots = h_info.get(self.spot_field, {}).get(self.content_field,
                                                           {})
        l_e = entity_spots
        s_e = set(h_info[self.spot_field].get(self.salience_field, []))
        test_label = [1 if e in s_e else -1 for e in l_e]
        l_label_org = [1 if label > 0 else -1 for label in test_label]
        # Associate label with eid.
        s_labels = dict(zip(l_e, l_label_org))

        l_kept_e, l_e_tf = self.get_top_k_e(l_e, self.max_e_per_d)
        l_label = [s_labels[e] for e in l_kept_e]
        ll_kept_feature = []
        if 'features' in entity_spots:
            # Associate features with eid.
            ll_feature = entity_spots.get('features', [[]] * len(l_e))
            h_e_feature = dict(zip(l_e, ll_feature))
            ll_kept_feature = [h_e_feature[e] for e in l_kept_e]
        h_res = {
            'mtx_e': l_kept_e,
            'mtx_score': l_e_tf,
            'ts_feature': ll_kept_feature,
            'label': l_label
        }
        return h_res

    def _parse_word(self, h_info):
        l_words = h_info.get(self.content_field, [])
        l_words, l_score = self.get_top_k_e(l_words, self.max_w_per_d)
        h_res = {
            'mtx_w': l_words,
            'mtx_w_score': l_score,
        }
        return h_res

    def _is_empty(self, data, dim=2):
        if dim == 2:
            return all([len(d) == 0 for d in data])
        if dim == 3:
            return all([len(d) == 0 for d in data]) \
                   or all([len(d) == 0 for row in data for d in row])

    def _padding(self, data, dim=2, default_value=0):
        if dim == 2:
            return self.two_d_padding(data, default_value)
        if dim == 3:
            return self.three_d_padding(data, default_value)
        raise NotImplementedError

    @classmethod
    def two_d_padding(cls, ll, default_value):
        n = max([len(l) for l in ll])
        for i in xrange(len(ll)):
            num_pad = n - len(ll[i])
            ll[i] += [default_value] * num_pad
        return ll

    @classmethod
    def three_d_padding(cls, lll, default_value):
        l_dim = [len(lll), 0, 0]
        for ll in lll:
            l_dim[1] = max(l_dim[1], len(ll))

            for l in ll:
                l_dim[2] = max(l_dim[2], len(l))

        for i in xrange(len(lll)):
            for j in xrange(len(lll[i])):
                lll[i][j] += [default_value] * (l_dim[2] - len(lll[i][j]))
            while len(lll[i]) < l_dim[1]:
                lll[i].append([default_value] * l_dim[2])
        return lll

    @classmethod
    def get_top_k_e(cls, l_term, max_number):
        h_e_tf = term2lm(l_term)
        l_e_tf = sorted(h_e_tf.items(), key=lambda item: -item[1])[:max_number]
        l_term = [item[0] for item in l_e_tf]
        z = float(sum([item[1] for item in l_e_tf]))
        l_w = [item[1] / z for item in l_e_tf]
        return l_term, l_w


"""
=============To be deprecated data i/o functions=============
"""


def padding(ll, filler):
    n = max([len(l) for l in ll])
    for i in xrange(len(ll)):
        ll[i] += [filler] * (n - len(ll[i]))
    return ll


def three_d_padding(lll, filler):
    l_dim = [len(lll), 0, 0]
    for ll in lll:
        l_dim[1] = max(l_dim[1], len(ll))
        for l in ll:
            l_dim[2] = max(l_dim[2], len(l))
    for i in xrange(len(lll)):
        for j in xrange(len(lll[i])):
            lll[i][j] += [filler] * (l_dim[2] - len(lll[i][j]))
        while len(lll[i]) < l_dim[1]:
            lll[i].append([filler] * l_dim[2])
    return lll


def get_top_k_e(l_e, max_e_per_d):
    h_e_tf = term2lm(l_e)
    l_e_tf = sorted(h_e_tf.items(), key=lambda item: -item[1])[:max_e_per_d]
    l_e = [item[0] for item in l_e_tf]
    z = float(sum([item[1] for item in l_e_tf]))
    l_w = [item[1] / z for item in l_e_tf]
    return l_e, l_w


def raw_io(l_line, num_features, spot_field=SPOT_FIELD,
           in_field=body_field, salience_field=abstract_field,
           salience_gold_field=salience_gold, max_e_per_d=200):
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


def _get_entity_info(entity_spots, abstract_spots, salience_gold_field,
                     max_e_per_d, num_features):
    l_e = entity_spots.get('entities', [])

    # Take label from salience field.
    if salience_gold_field in entity_spots:
        test_label = entity_spots[salience_gold_field]
    else:
        # backward compatibility
        s_e = set(abstract_spots.get('entities', []))
        test_label = [1 if e in s_e else -1 for e in l_e]

    l_label_org = [1 if label == 1 else -1 for label in test_label]
    # Associate label with eid.
    s_labels = dict(zip(l_e, l_label_org))

    # Associate features with eid.
    ll_feature = entity_spots.get('features', [[]] * len(l_e))
    s_features = dict(zip(l_e, ll_feature))

    l_e, l_w = get_top_k_e(l_e, max_e_per_d)
    l_label = [s_labels[e] for e in l_e]

    if num_features:
        ll_feature = [s_features[e][:num_features] for e in l_e]
        return l_e, l_label, ll_feature
    else:
        # Mention count is appended to the first position in feature array.
        l_e_tf = [s_features[e][0] for e in l_e]
        z = float(sum(l_e_tf))
        l_w = [tf / z for tf in l_e_tf]
        return l_e, l_label, l_w


def _offset_hash(l_e, offset):
    return [e + offset for e in l_e]


def feature_io(l_line, num_features, spot_field=SPOT_FIELD, in_field=body_field,
               salience_field=abstract_field, salience_gold_field=salience_gold,
               max_e_per_d=200):
    """
    io with pre-filtered entity list and feature matrices.
    fall back to raw io output if num_features = 0
    """
    ll_e = []
    lll_feature = []
    ll_label = []
    f_dim = 0

    for line in l_line:
        h = json.loads(line)
        entity_spots = h[spot_field].get(in_field, {})
        abstract_spots = h[spot_field].get(salience_field, {})

        l_e, l_label, ll_feature = _get_entity_info(entity_spots,
                                                    abstract_spots,
                                                    salience_gold_field,
                                                    max_e_per_d, num_features)

        if not l_e:
            continue

        if num_features and ll_feature:
            f_dim = max(f_dim, len(ll_feature[0]))

        ll_e.append(l_e)
        ll_label.append(l_label)
        lll_feature.append(ll_feature)

    ll_e = padding(ll_e, 0)
    ll_label = padding(ll_label, 0)

    if num_features:
        lll_feature = padding(lll_feature, [0] * f_dim)
    else:
        lll_feature = padding(lll_feature, 0)

    m_e = Variable(torch.LongTensor(ll_e)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_e))
    m_label = Variable(torch.FloatTensor(ll_label)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_label))
    ts_feature = Variable(torch.FloatTensor(lll_feature)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(lll_feature))

    if num_features:
        h_packed_data = {
            "mtx_e": m_e,
            "ts_feature": ts_feature
        }
    else:
        h_packed_data = {
            "mtx_e": m_e,
            "mtx_score": ts_feature
        }
    return h_packed_data, m_label


def uw_io(l_line, spot_field=SPOT_FIELD,
          in_field=body_field, salience_field=abstract_field, max_e_per_d=200):
    """
    prepare local words around each entity's location
    :param l_line: hashed data with loc fields
    :param spot_field: together with in_field, to get
    :param in_field:
    :param salience_field: to get label
    :param max_e_per_d: Number of maximum entities allowed.
    :return:
    """
    sent_len = 10
    max_sent_cnt = 0
    max_sent_allowed = 5
    ll_e = []
    ll_label = []
    lll_sent = []  #

    for line in l_line:
        h = json.loads(line)
        packed = h[spot_field].get(in_field, {})
        l_e = packed.get('entities', [])
        ll_loc = packed.get('loc', [])
        l_words = h[in_field]
        ll_sent = [
            _form_local_context(l_loc[:max_sent_allowed], l_words, sent_len)
            for l_loc in ll_loc]

        this_max_sent_cnt = max([len(l_sent) for l_sent in ll_sent])
        max_sent_cnt = max(max_sent_cnt, this_max_sent_cnt)

        s_salient_e = set(
            h[spot_field].get(salience_field, {}).get('entities', []))
        l_label = [1 if e in s_salient_e else -1 for e in l_e]

        ll_e.append(l_e)
        ll_label.append(l_label)
        lll_sent.append(ll_sent)

    for d_p in xrange(len(lll_sent)):
        for e_p in xrange(len(lll_sent[d_p])):
            lll_sent[d_p][e_p] += [[0] * sent_len] * (
                    max_sent_cnt - len(lll_sent[d_p][e_p]))

    ll_e = padding(ll_e, 0)
    ll_label = padding(ll_label, 0)

    l_empty_e_sent = [[0] * sent_len] * max_sent_cnt
    lll_sent = padding(lll_sent, l_empty_e_sent)

    m_e = Variable(torch.LongTensor(ll_e)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_e))
    m_label = Variable(torch.FloatTensor(ll_label)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_label))
    ts_local_context = Variable(torch.LongTensor(lll_sent)).cuda() \
        if use_cuda else Variable(torch.LongTensor(lll_sent))
    h_packed_data = {
        "mtx_e": m_e,
        "ts_local_context": ts_local_context
    }
    return h_packed_data, m_label


def _form_local_context(l_loc, l_words, sent_len):
    """
    get local UW words
    :param l_loc: positions to get UW words
    :param l_words: content
    :param sent_len: length of UW
    :return:
    """
    l_sent = []
    for st, ed in l_loc:
        uw_st = max(st - sent_len / 2, 0)
        uw_st = min(uw_st, len(l_words) - sent_len)
        sent = l_words[uw_st: uw_st + sent_len]
        sent += [0] * (sent_len - len(sent))
        l_sent.append(sent)
    return l_sent


def duet_io(l_line, spot_field=SPOT_FIELD,
            in_field=body_field, salience_field=abstract_field,
            max_e_per_d=200, max_w_per_d=500):
    """
    add the whole document's word sequence as mtx_w
    :param l_line:
    :param spot_field:
    :param in_field:
    :param salience_field:
    :param max_e_per_d:
    :param max_w_per_d:
    :return:
    """
    ll_e = []
    ll_score = []
    ll_words = []
    ll_word_score = []
    ll_label = []
    for line in l_line:
        h = json.loads(line)
        l_e = h[spot_field].get(in_field, [])
        l_e, l_score = get_top_k_e(l_e, max_e_per_d)
        s_salient_e = set(h[spot_field].get(salience_field, []))
        l_label = [1 if e in s_salient_e else -1 for e in l_e]
        ll_e.append(l_e)
        ll_score.append(l_score)
        ll_label.append(l_label)
        l_word = h.get(in_field, [])
        l_word, l_word_score = get_top_k_e(l_word, max_w_per_d)
        ll_words.append(l_word)
        ll_word_score.append(l_word_score)

    ll_e = padding(ll_e, 0)
    ll_score = padding(ll_score, 0)
    ll_label = padding(ll_label, 0)
    ll_words = padding(ll_words, 0)
    ll_word_score = padding(ll_word_score, 0)
    m_e = Variable(torch.LongTensor(ll_e)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_e))
    m_w = Variable(torch.FloatTensor(ll_score)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_score))
    m_label = Variable(torch.FloatTensor(ll_label)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_label))
    m_word = Variable(torch.LongTensor(ll_words)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_words))
    m_word_score = Variable(torch.FloatTensor(ll_word_score)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_word_score))
    h_packed_data = {
        "mtx_e": m_e,
        "mtx_score": m_w,
        "mtx_w": m_word,
        "mtx_w_score": m_word_score,
    }
    return h_packed_data, m_label


def adj_edge_io(
        l_line, spot_field=SPOT_FIELD,
        in_field=body_field, salience_field=abstract_field,
        max_e_per_d=200
):
    """
    convert data to the input for the model
    """
    ll_e = []
    ll_w = []
    ll_label = []
    lll_e_distance = []
    for line in l_line:
        h = json.loads(line)
        l_seq_e = h[spot_field].get(in_field, [])
        l_e, l_w = get_top_k_e(l_seq_e, max_e_per_d)
        ll_e_distance = _form_distance_mtx(l_seq_e, l_e)
        s_salient_e = set(h[spot_field].get(salience_field, []))
        l_label = [1 if e in s_salient_e else -1 for e in l_e]
        ll_e.append(l_e)
        ll_w.append(l_w)
        ll_label.append(l_label)
        lll_e_distance.append(ll_e_distance)

    ll_e = padding(ll_e, 0)
    ll_w = padding(ll_w, 0)
    ll_label = padding(ll_label, 0)
    lll_e_distance = three_d_padding(lll_e_distance, -1)
    m_e = Variable(torch.LongTensor(ll_e)).cuda() \
        if use_cuda else Variable(torch.LongTensor(ll_e))
    m_w = Variable(torch.FloatTensor(ll_w)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_w))
    m_label = Variable(torch.FloatTensor(ll_label)).cuda() \
        if use_cuda else Variable(torch.FloatTensor(ll_label))
    ts_e_distance = Variable(
        torch.LongTensor(lll_e_distance)).cuda() \
        if use_cuda else Variable(torch.LongTensor(lll_e_distance))
    h_packed_data = {
        "mtx_e": m_e,
        "mtx_score": m_w,
        'ts_distance': ts_e_distance,
    }
    return h_packed_data, m_label


def _form_distance_mtx(l_seq_e, l_e):
    h_e_p = dict(zip(l_e, range(len(l_e))))
    ll_distance = []
    for i in xrange(len(l_e)):
        ll_distance.append([-1] * len(l_e))
        ll_distance[i][i] = 0

    for i in xrange(len(l_seq_e)):
        if l_seq_e[i] not in h_e_p:
            continue
        p_i = h_e_p[l_seq_e[i]]
        for j in xrange(i + 1, len(l_seq_e)):
            if l_seq_e[j] not in h_e_p:
                continue
            p_j = h_e_p[l_seq_e[j]]
            ll_distance[p_i][p_j] = min(j - i, ll_distance[p_i][p_j]) if \
                ll_distance[p_i][p_j] != -1 \
                else j - i
    return ll_distance


if __name__ == '__main__':
    """
    unit test tbd
    """
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    from traitlets.config import Configurable
    from traitlets import (
        Unicode,
        List,
        Int
    )

    set_basic_log()


    class IOTester(Configurable):
        in_name = Unicode(help='in data test').tag(config=True)
        io_func = Unicode('uw', help='io function to test').tag(config=True)
