"""
convert external semantics to np arrays
semantics:
    entity description
    entity NLSS
    entity triples

input:
    e id hash dict (from convert_vocab_hash_and_emb_mtx.py)
    word id hash dict
    type id hash dict
    semantics in json format
        id: str
        desp: str
        nlss: []
        rdf: []
output:
    a hashed arrary for each given semantics
"""

import sys
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
)
import logging
import json
import numpy as np
import pickle
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


class ExtSemanticPrep(Configurable):
    e_id_hash_in = Unicode(help='entity id hash in').tag(config=True)
    word_id_hash_in = Unicode(help='word id hash in').tag(config=True)
    semantic_json_in = Unicode(help='semantic json hash in').tag(config=True)
    out_name = Unicode(help='output pre').tag(config=True)
    max_desp_len = Int(100, help='maximum description length').tag(config=True)
    max_nlss_len = Int(20, help='maximum NLSS len').tag(config=True)
    max_nlss_nb = Int(20, help='number of nlss to use').tag(config=True)
    max_rdf_nb = Int(100, help='number of rdf triples to use').tag(config=True)

    def __init__(self, **kwargs):
        super(ExtSemanticPrep, self).__init__(**kwargs)

        self.h_e_id = dict()
        self.h_w_id = dict()
        self._load_ids()


    def process(self):
        ts_desp, ts_rdf, ts_nlss = self._load_semantics()
        if ts_desp:
            logging.info('saving desp ts %s', json.dumps(ts_desp.shape))
            np.save(open(self.out_name + '.desp.npy', 'wb'), ts_desp)

        if ts_rdf:
            logging.info('saving rdf ts %s', json.dumps(ts_rdf.shape))
            np.save(open(self.out_name + '.rdf.npy', 'wb'), ts_rdf)

        if ts_nlss:
            logging.info('saving nlss ts %s', json.dumps(ts_nlss.shape))
            np.save(open(self.out_name + '.nlss.npy', 'wb'), ts_nlss)
        logging.info('external semantics hashed and dumped to [%s]', self.out_name)
        return

    def _load_ids(self):
        logging.info('loading e id [%s]...', self.e_id_hash_in)
        self.h_e_id = pickle.load(open(self.e_id_hash_in))
        logging.info('%d entities', len(self.h_e_id))
        logging.info('loading w id [%s]...', self.word_id_hash_in)
        self.h_w_id = pickle.load(open(self.word_id_hash_in))
        logging.info('%d words', len(self.h_w_id))

    def _load_semantics(self):
        ts_desp, ts_rdf, ts_nlss = None, None, None
        if not self.semantic_json_in:
            return ts_desp, ts_rdf, ts_nlss

        has_desp = False
        has_rdf = False
        has_nlss = False

        nb_e = len(self.h_e_id)
        logging.info('initializing zero starts')
        ts_desp = np.zeros((nb_e, self.max_desp_len))
        ts_rdf = np.zeros((nb_e, self.max_rdf_nb, 2))
        ts_nlss = np.zeros((nb_e, self.max_nlss_nb, self.max_nlss_len))

        e_cnt = 0
        for p, line in enumerate(open(self.semantic_json_in)):
            if not p % 1000:
                logging.info('loaded [%d] json h [%d] targets', p, e_cnt)

            h = json.loads(p)
            e_id = h['id']
            if e_id not in self.h_e_id:
                continue
            e_cnt += 1
            e_pos = self.h_e_id[e_id]

            '''
            only supporting desp for now
            '''
            if 'desp' in h:
                has_desp = True
                desp = h['desp'].lower()
                l = [self.h_w_id.get(w, 0) for w in desp.split()][:self.max_desp_len]
                ts_desp[e_pos][:len(l)] = np.array(l)

        logging.info('[%d/%d] json semantic processed, has_desp:%s, has_rdf:%s, has_nlss:%s',
                     e_cnt, nb_e, json.dumps(has_desp), json.dumps(has_rdf), json.dumps(has_nlss))

        if not has_desp:
            ts_desp = None
        if not has_rdf:
            ts_rdf = None
        if not has_nlss:
            ts_nlss = None

        return ts_desp, ts_rdf, ts_nlss


if __name__ == '__main__':
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "1 para, config"
        ExtSemanticPrep.class_print_help()

    preper = ExtSemanticPrep(config=load_py_config(sys.argv[1]))
    preper.process()






