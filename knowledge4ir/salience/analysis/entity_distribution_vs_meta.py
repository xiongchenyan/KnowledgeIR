"""
1/2/2018
the distribution of entities
    in gold
    in predicted (top 5)
VS
    meta:
        entity type (TBD 1/2/2018)
        frequency

input:
    predicted json
    meta dicts
"""

import json
from traitlets.config import Configurable
from traitlets import (
    Int,
    Unicode,
    List,
)
import logging
import numpy as np
import math


class EntityDistVSMeta(Configurable):
    df_meta_in = Unicode().tag(config=True)
    nb_bin = Int(5).tag(config=True)
    content_field = Unicode('bodyText').tag(config=True)
    top_k = Int(5).tag(config=True)

    def __init__(self, **kwargs):
        super(EntityDistVSMeta, self).__init__(**kwargs)
        self.h_e_df = dict()
        self.h_e_df_bin = dict()
        if self.df_meta_in:
            self.h_e_df = json.load(open(self.df_meta_in))
            logging.info('[%s] df meta loaded', self.df_meta_in)
            self.h_e_df_bin = self._bin_e_by_value(self.h_e_df)

    def _bin_e_by_value(self, h_e_value):
        l_item = h_e_value.items()
        l_item.sort(key=lambda item: item[1])
        l_e = [item[0] for item in l_item]
        bin_width = math.ceil(len(l_item) / float(self.nb_bin))
        st, ed = 0, bin_width
        h_e_bin = dict()
        bin_number = 0
        while st < len(l_e):
            l_this_e = list(l_e[st: ed])
            h_e_bin.update(dict(zip(l_this_e, [bin_number] * len(l_this_e))))
            st = ed
            ed += bin_width
            bin_number += 1
        return h_e_bin

    def _culminate_one_doc(self, h_doc_info, h_e_bin):
        l_gold_bin_cnt = [0] * self.nb_bin
        l_top_bin_cnt = [0] * self.nb_bin

        h_ana = h_doc_info.get('spot', {}).get(self.content_field, {})
        l_e = h_ana['entities']
        l_score = h_ana['predict']
        l_label = h_ana['salience']

        l_gold_e = [e for e, label in zip(l_e, l_label) if label > 0]
        l_top_e = [item[0] for item in sorted(zip(l_e, l_score), key=lambda item: -item[1])[:self.top_k]]

        for e in l_gold_e:
            b = h_e_bin[e]
            l_gold_bin_cnt[b] += 1

        for e in l_top_e:
            b = h_e_bin[e]
            l_top_bin_cnt[b] += 1

        return l_gold_bin_cnt, l_top_bin_cnt

    def process(self, in_name, out_name):

        l_gold_df_bin = np.array([0] * self.nb_bin)
        l_pre_df_bin = np.array([0] * self.nb_bin)

        for p, line in enumerate(open(in_name)):
            if not p % 1000:
                logging.info('processed [%d] doc', p)

            h_doc = json.loads(line)
            l_this_df_gold_bin, l_this_df_top_bin = self._culminate_one_doc(h_doc, self.h_e_df_bin)
            l_gold_df_bin += np.array(l_this_df_gold_bin)
            l_pre_df_bin += np.array(l_this_df_top_bin)
        logging.info('document processed')
        l_gold_df_bin = l_gold_df_bin.tolist()
        l_pre_df_bin = l_pre_df_bin.tolist()

        h_res = {
            'gold_df': l_gold_df_bin,
            'pre_df': l_pre_df_bin,
        }
        json.dump(h_res, open(out_name, 'w'), indent=1)
        logging.info('finished')
        return



if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import load_py_config, set_basic_log
    set_basic_log()

    if 4 != len(sys.argv):
        print "study the entity distributions in labels and predictions"
        print "3 para: config + predicted json + out name"
        sys.exit(-1)

    ana = EntityDistVSMeta(config=load_py_config(sys.argv[1]))
    ana.process(*sys.argv[1:])


