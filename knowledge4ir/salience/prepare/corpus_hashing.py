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


class CorpusHasher(Configurable):
    word_id_pickle_in = Unicode(help='pickle of word  id').tag(config=True)
    entity_id_pickle_in = Unicode(help='pickle of entity id').tag(config=True)
    corpus_in = Unicode(help='input').tag(config=True)
    out_name = Unicode().tag(config=True)
    with_feature = Bool(False, help='whether load feature, or just frequency').tag(config=True)
    max_e_per_d = Int(200, help="top k frequent entities to use per doc").tag(config=True)
    with_position = Bool(False, help='whether add position').tag(config=True)
    max_position_per_e = Int(20, help='max loc per e to keep').tag(config=True)

    def __init__(self, **kwargs):
        super(CorpusHasher, self).__init__(**kwargs)
        self.h_word_id = pickle.load(open(self.word_id_pickle_in))
        self.h_entity_id = pickle.load(open(self.entity_id_pickle_in))
        logging.info('loaded [%d] word id [%d] entity id',
                     len(self.h_word_id), len(self.h_entity_id))

    def hash_per_info(self, h_info):
        h_hashed = dict()
        l_field = [field for field in h_info.keys() if field not in {'qid', 'docno', 'spot'}]
        for field in l_field:
            text = h_info[field]
            l_w = text.lower().split()
            l_w_id = [self.h_word_id.get(w, 0) for w in l_w]
            h_hashed[field] = l_w_id
        for key in ['qid', 'docno']:
            if key in h_info:
                h_hashed[key] = h_info[key]

        h_hashed['spot'] = dict()
        for field, l_ana in h_info['spot'].items():
            l_ana_id = [self.h_entity_id.get(ana['entities'][0]['id'], 0)
                        for ana in l_ana]
            if not l_ana_id:
                this_field_data = {
                    "entities": [],
                    "features": [],
                }
                if self.with_position:
                    this_field_data['loc'] = []
                h_hashed['spot'][field] = this_field_data
                continue
            l_id_tf = term2lm([id for id in l_ana_id if id != 0]).items()
            l_id_tf.sort(key=lambda item: -item[1])
            l_id_tf = l_id_tf[:self.max_e_per_d]
            l_id = [item[0] for item in l_id_tf]
            ll_feature = []
            for e_id, tf in l_id_tf:
                ll_feature.append([tf])

            if self.with_feature:
                ll_feature = self._add_node_features(l_ana, l_id_tf, ll_feature)

            this_field_data = {
                "entities": l_id,
                "features": ll_feature
            }
            if self.with_position:
                ll_position = self._add_entity_loc(l_ana, l_id_tf)
                this_field_data['loc'] = ll_position
            h_hashed['spot'][field] = this_field_data

        return h_hashed

    def process(self):
        out = open(self.out_name, 'w')
        for p, line in enumerate(open(self.corpus_in)):
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

    def _add_node_features(self, l_ana, l_id_tf, ll_feature):
        assert len(l_id_tf) == ll_feature
        l_ana_id = [self.h_entity_id.get(ana['entities'][0]['id'], 0)
                    for ana in l_ana]
        ll_e_features = [ana['entities'][0].get('feature', {}).get('featureArray', [])
                         for ana in l_ana]
        h_e_id_feature = dict(zip(l_ana_id, ll_e_features))
        feature_dim = max([len(l_f) for l_f in ll_e_features])
        for p in xrange(len(l_id_tf)):
            e_id, tf = l_id_tf[p]
            l_feature = h_e_id_feature[e_id]
            l_feature += [0] * (feature_dim - len(l_feature))
            ll_feature[p] += l_feature
        return ll_feature

    def _add_entity_loc(self, l_ana, l_id_tf):
        l_ana_id = [self.h_entity_id.get(ana['entities'][0]['id'], 0)
                    for ana in l_ana]
        l_ana_loc = [ana['loc'] for ana in l_ana]
        h_id_loc = {}
        for e_id, loc in zip(l_ana_id, l_ana_loc):
            l_loc = h_id_loc.get(e_id, [])
            if len(l_loc) >= self.max_position_per_e:
                continue
            l_loc.append(loc)
            h_id_loc[e_id] = l_loc

        ll_loc = []
        for e_id, _ in l_id_tf:
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
