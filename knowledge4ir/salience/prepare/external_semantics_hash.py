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
    Bool,
)
import logging
import gzip
import json
import numpy as np
import pickle
from knowledge4ir.utils import (
    tokenize_and_remove_punctuation,
)
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
    compressed_input = Bool(False, help='whether is compressed input').tag(config=True)

    def __init__(self, **kwargs):
        super(ExtSemanticPrep, self).__init__(**kwargs)

        self.h_e_id = dict()
        self.h_w_id = dict()
        self._load_ids()

    def process(self):
        ts_desp, ts_rdf, ts_nlss = self._load_semantics()
        if ts_desp is not None:
            logging.info('saving desp ts %s', json.dumps(ts_desp.shape))
            np.save(open(self.out_name + '.desp.npy', 'wb'), ts_desp)

        if ts_rdf is not None:
            logging.info('saving rdf ts %s', json.dumps(ts_rdf.shape))
            np.save(open(self.out_name + '.rdf.npy', 'wb'), ts_rdf)

        if ts_nlss is not None:
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

        nb_e = len(self.h_e_id)
        logging.info('initializing zero starts')
        ts_desp = np.zeros((nb_e, self.max_desp_len), dtype=np.int)
        ts_rdf = np.zeros((nb_e, self.max_rdf_nb, 2), dtype=np.int)
        ts_nlss = np.zeros((nb_e, self.max_nlss_nb, self.max_nlss_len), dtype=np.int)
        l_nlss_idx = [0] * nb_e
        desp_cnt = 0
        rdf_cnt = 0
        nlss_cnt = 0
        f_in = gzip.open(self.semantic_json_in) if self.compressed_input else open(self.semantic_json_in)
        for p, line in enumerate(f_in):
            if not p % 1000:
                logging.info('loaded [%d] json lines, desp|rdf|nlss cnt: [%d|%d|%d]',
                             p, desp_cnt, rdf_cnt, nlss_cnt)
            h = json.loads(line)
            pos, l_desp = self._fetch_desp(h)
            if pos is not None:
                desp_cnt += 1
                ts_desp[pos][:len(l_desp)] = np.array(l_desp)
            l_pos, ll_nlss_words = self._fetch_nlss(h)
            if l_pos is not None:
                for pos, l_nlss_words in zip(l_pos, ll_nlss_words):
                    while l_nlss_idx[pos] <= self.max_nlss_nb:
                        ts_nlss[pos][l_nlss_idx[pos]][:len(l_nlss_words)] = np.array(l_nlss_words)
                        l_nlss_idx[pos] += 1
                        nlss_cnt += 1

        logging.info('total nb_desp:%d, nb_rdf:%d, nb_nlss:%d, target e:%d',
                     desp_cnt, rdf_cnt, nlss_cnt, nb_e)
        if not desp_cnt:
            ts_desp = None
        if not rdf_cnt:
            ts_rdf = None
        if not nlss_cnt:
            ts_nlss = None
        f_in.close()
        return ts_desp, ts_rdf, ts_nlss

    def _fetch_desp(self, h):
        e_id = h.get('id', None)
        if e_id not in self.h_e_id:
            return None, None
        e_pos = self.h_e_id[e_id]

        if 'desp' in h:
            has_desp = True
            desp = h['desp'].lower()
            l_desp = [self.h_w_id.get(w, 0) for w in desp.split()][:self.max_desp_len]
            return e_pos, l_desp
        return None, None

    def _fetch_nlss(self, h):
        if 'supports' not in h:
            return None, None
        l_pos, ll_nlss_words = [], []
        for support_info in h['supports']:
            e_id = support_info['id']
            l_sent = support_info['sentences']
            if e_id not in self.h_e_id:
                continue
            e_pos = self.h_e_id[e_id]
            for sent in l_sent:
                l_words = tokenize_and_remove_punctuation(sent.lower())
                l_w_id = [self.h_w_id.get(w, 0) for w in l_words][:self.max_nlss_len]
                l_pos.append(e_pos)
                ll_nlss_words.append(l_w_id)
        return l_pos, ll_nlss_words


if __name__ == '__main__':
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "1 para, config"
        ExtSemanticPrep.class_print_help()
        sys.exit(-1)

    prep = ExtSemanticPrep(config=load_py_config(sys.argv[1]))
    prep.process()






