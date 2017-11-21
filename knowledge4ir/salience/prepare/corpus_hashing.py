"""
hash the json format training and testing corpus
"""

import json
import logging
from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
    Bool,
)
from knowledge4ir.utils import term2lm, body_field, title_field
import pickle
import numpy as np
import gzip
from collections import defaultdict
from itertools import chain

UNK_TOKEN = "<unk>"


def get_lookup():
    lookup = defaultdict(lambda: len(lookup))
    unk = lookup[UNK_TOKEN]
    return unk, lookup


def fix_lookup(lookup):
    return defaultdict(lambda: UNK_TOKEN, lookup)


def load_frame_names(frame_file):
    unk, f2id = get_lookup()
    with open(frame_file) as f:
        for line in f:
            frame = line.strip()
            _ = f2id[frame]
    f2id = fix_lookup(f2id)
    return f2id


class CorpusHasher(Configurable):
    word_id_pickle_in = Unicode(help='pickle of word id').tag(config=True)
    entity_id_pickle_in = Unicode(help='pickle of entity id').tag(config=True)
    corpus_in = Unicode(help='input').tag(config=True)
    out_name = Unicode().tag(config=True)
    with_feature = Bool(False,
                        help='whether load feature, or just frequency').tag(
        config=True)
    max_e_per_d = Int(200, help="top k frequent entities to use per doc").tag(
        config=True)
    with_position = Bool(False, help='whether add position').tag(config=True)
    max_position_per_e = Int(20, help='max loc per e to keep').tag(config=True)
    hash_events = Bool(False, help="whether to hash evnet information").tag(
        config=True)
    frame_name_file = Unicode(help="file containing possible frame names").tag(
        config=True)

    def __init__(self, **kwargs):
        super(CorpusHasher, self).__init__(**kwargs)
        self.h_frame_id = load_frame_names(self.frame_name_file)
        self.h_word_id = pickle.load(open(self.word_id_pickle_in))
        self.h_entity_id = pickle.load(open(self.entity_id_pickle_in))

        self.sparse_feature_dicts = {}

        logging.info('loaded [%d] word ids, [%d] entity ids, [%d] frames',
                     len(self.h_word_id), len(self.h_entity_id),
                     len(self.h_frame_id))

    def _hash_spots(self, h_info, h_hashed):
        h_hashed['spot'] = dict()
        for field, l_ana in h_info['spot'].items():
            l_ana_id = [self.h_entity_id.get(ana['id'], 0) for ana in l_ana]
            if not l_ana_id:
                this_field_data = {
                    "entities": [],
                    "features": [],
                    "salience": []
                }
                if self.with_position:
                    this_field_data['loc'] = []
                h_hashed['spot'][field] = this_field_data
                continue

            l_id_tf = term2lm([eid for eid in l_ana_id if eid != 0]).items()
            l_id_tf.sort(key=lambda item: -item[1])
            l_id_tf = l_id_tf[:self.max_e_per_d]
            l_id = [item[0] for item in l_id_tf]

            ll_feature = [[tf] for eid, tf in l_id_tf]
            if self.with_feature:
                ll_feature = self._add_node_features(l_ana, l_id, ll_feature)

            l_salience = self._get_node_salience(l_ana, l_id)

            this_field_data = {
                "entities": l_id,
                "features": ll_feature,
                "salience": l_salience
            }
            if self.with_position:
                ll_position = self._add_entity_loc(l_ana, l_id)
                this_field_data['loc'] = ll_position
            h_hashed['spot'][field] = this_field_data

    def _hash_events(self, h_info, h_hashed):
        h_hashed['event'] = dict()
        for field, l_ana in h_info['event'].items():
            event_frames = [self.h_frame_id.get(ana['frame_name'], 0)
                            for ana in l_ana]
            if not event_frames:
                this_field_data = {
                    "frames": [],
                    "features": [],
                }
                if self.with_position:
                    this_field_data['loc'] = []
                h_hashed['event'][field] = this_field_data

            ll_feature = [[] for _ in l_ana]
            ll_sparse_features = [[] for _ in l_ana]

            if self.with_feature:
                ll_sparse_features, ll_feature = \
                    self._add_event_features(l_ana, ll_feature)

            l_salience = self._get_event_salience(l_ana)

            this_field_data = {
                "frames": event_frames,
                "features": ll_feature,
                "sparse_features": ll_sparse_features,
                "salience": l_salience
            }
            if self.with_position:
                ll_position = self._add_event_loc(l_ana)
                this_field_data['loc'] = ll_position
            h_hashed['event'][field] = this_field_data

    def hash_per_info(self, h_info):
        h_hashed = dict()
        l_field = [field for field in h_info.keys() if
                   field not in {'qid', 'docno', 'spot', 'event'}]
        for field in l_field:
            text = h_info[field]
            l_w = text.lower().split()
            l_w_id = [self.h_word_id.get(w, 0) for w in l_w]
            h_hashed[field] = l_w_id
        for key in ['qid', 'docno']:
            if key in h_info:
                h_hashed[key] = h_info[key]

        # TODO: add sanity check.
        self._hash_spots(h_info, h_hashed)
        if self.hash_events:
            self._hash_events(h_info, h_hashed)

        return h_hashed

    def process(self):
        out = open(self.out_name, 'w')
        open_func = gzip.open if self.corpus_in.endswith("gz") else open
        with open_func(self.corpus_in) as in_f:
            for p, line in enumerate(in_f):
                if not p % 1000:
                    logging.info('processing [%d] lines', p)
                h_hashed = self.hash_per_info(json.loads(line))
                if not h_hashed['spot'][body_field]['entities']:
                    continue
                if not h_hashed['spot']['abstract']['entities']:
                    continue
                print >> out, json.dumps(h_hashed)

        out.close()
        logging.info('finished')
        return

    def _add_event_features(self, l_ana, ll_feature):
        assert len(l_ana) == len(ll_feature)
        if len(l_ana) == 0:
            return [], []

        ll_e_features = [
            ana.get('feature', {}).get('featureArray', []) for ana in l_ana
        ]

        l_sparse_features = [
            ana.get('feature', {}).get('sparseFeatureArray', [])
            for ana in l_ana
        ]

        sparse_feature_names = set(chain.from_iterable(
            ana.get('feature', {}).get('sparseFeatureName', [])
            for ana in l_ana
        ))

        # Create a lookup for each sparse feature type.
        for n in sparse_feature_names:
            # unk will always be 0.
            unk, self.sparse_feature_dicts[n] = get_lookup()

        sparse_feature_dim = len(sparse_feature_names)

        ll_sparse_features = [[] for _ in l_ana]
        for p, features in enumerate(l_sparse_features):
            l_fids = []
            for f in features:
                fname, fvalue = f.split("_", 1)
                fid = self.sparse_feature_dicts[fname][fvalue]
                l_fids.append(fid)
            l_fids += [0] * (sparse_feature_dim - len(l_fids))
            ll_sparse_features[p] = l_fids

        feature_dim = max([len(l_f) for l_f in ll_e_features])

        for p, l_feature in enumerate(ll_e_features):
            l_feature += [0] * (feature_dim - len(l_feature))
            ll_feature[p] += l_feature

        return ll_sparse_features, ll_feature

    def _get_node_salience(self, l_ana, valid_ids):
        raw_salience = [
            ana.get('salience', 0) for ana in l_ana
        ]
        l_ana_id = [self.h_entity_id.get(ana['id'], 0) for ana in l_ana]
        h_salience = dict(zip(l_ana_id, raw_salience))

        l_salience = [0] * len(valid_ids)
        for p, e_id in enumerate(valid_ids):
            l_salience[p] = h_salience[e_id]
        return l_salience

    def _get_event_salience(self, l_ana):
        return [ana.get('salience', 0) for ana in l_ana]

    def _add_node_features(self, l_ana, valid_ids, ll_feature):
        assert len(valid_ids) == len(ll_feature)
        l_ana_id = [self.h_entity_id.get(ana['id'], 0) for ana in l_ana]
        ll_e_features = [
            ana.get('feature', {}).get('featureArray', []) for ana in l_ana
        ]
        h_e_id_feature = dict(zip(l_ana_id, ll_e_features))
        feature_dim = max([len(l_f) for l_f in ll_e_features])
        for p, e_id in enumerate(valid_ids):
            l_feature = h_e_id_feature[e_id]
            l_feature += [0] * (feature_dim - len(l_feature))
            ll_feature[p] += l_feature

        return ll_feature

    def _add_event_loc(self, l_ana):
        # There is no coreference for event, each event have one single loc.
        ll_loc = [[ana['loc']] for ana in l_ana]
        return ll_loc

    def _add_entity_loc(self, l_ana, valid_ids):
        l_ana_id = [self.h_entity_id.get(ana['id'], 0) for ana in l_ana]
        l_ana_loc = [ana['loc'] for ana in l_ana]
        h_id_loc = {}

        for e_id, loc in zip(l_ana_id, l_ana_loc):
            l_loc = h_id_loc.get(e_id, [])
            if len(l_loc) >= self.max_position_per_e:
                continue
            l_loc.append(loc)
            h_id_loc[e_id] = l_loc

        ll_loc = []
        for e_id in valid_ids:
            ll_loc.append(h_id_loc[e_id])

        return ll_loc


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log
    )

    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "hashing corpus, 1 para, config:"
        CorpusHasher.class_print_help()
        sys.exit(-1)

    hasher = CorpusHasher(config=load_py_config(sys.argv[1]))
    hasher.process()
