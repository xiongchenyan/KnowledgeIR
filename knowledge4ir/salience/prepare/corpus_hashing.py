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
    l_target_field = List(Unicode, default_value=[body_field]).tag(config=True)
    max_e_per_d = Int(200, help="top k frequent entities to use per doc").tag(config=True)

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
            if field not in self.l_target_field:
                continue
            l_ana_id = [self.h_entity_id.get(ana['entities'][0]['id'], 0)
                        for ana in l_ana]
            ll_e_features = [ana['entities'][0].get('feature', {}).get('featureArray', [])
                             for ana in l_ana]
            h_e_id_feature = dict(zip(l_ana_id, ll_e_features))
            l_id_tf = term2lm([id for id in l_ana_id if id != 0]).items()
            l_id_tf.sort(key=lambda item: -item[1])
            l_id_tf = l_id_tf[:self.max_e_per_d]
            l_id = [item[0] for item in l_id_tf]
            ll_feature = []
            feature_dim = max([len(l_f) for l_f in ll_e_features])
            if self.with_feature:
                if not feature_dim:
                    logging.error('doc [%s] feature empty', h_hashed['docno'])
                assert feature_dim

            for e_id, tf in l_id_tf:
                l_feature = h_e_id_feature[e_id]
                l_feature += [0] * (feature_dim - len(l_feature))
                l_feature = [tf] + l_feature
                ll_feature.append(l_feature)
            this_field_data = {
                "entities": l_id,
                "features": ll_feature
            }

            h_hashed['spot'][field] = this_field_data
        return h_hashed

    def process(self):
        out = open(self.out_name, 'w')
        for p, line in enumerate(open(self.corpus_in)):
            if not p % 1000:
                logging.info('processing [%d] lines', p)
            h_hashed = self.hash_per_info(json.loads(line))
            print >> out, json.dumps(h_hashed)

        out.close()
        logging.info('finished')
        return


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
