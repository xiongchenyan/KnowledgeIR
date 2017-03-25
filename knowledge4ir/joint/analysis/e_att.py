"""
analysis e attention weights

input:
    intermediate results of model's output
        each line's ['meta']['e_att_score'] ->
            [a sf form's[(e, att score),()],
output:
    stats of attention weights (normalized to sum to 1)
        distribution of the first e's rank
        fraction of weights of the first e
        att weights's distribution (in 10 bins 0-1)
"""

import numpy as np
import json
import sys
import logging
from traitlets.config import Configurable
from traitlets import (
    Unicode
)
import math


class EAttAna(Configurable):
    in_name = Unicode(help='input intermediate res').tag(config=True)
    out_name = Unicode(help='out name').tag(config=True)

    def __init__(self, **kwargs):
        super(EAttAna, self).__init__(**kwargs)
        self.l_e_att_mtx = self._load_e_att_mtx()
        self.l_e_att_mtx = self._normalize_e_att_weights(self.l_e_att_mtx)

    def _load_e_att_mtx(self):
        l_e_att_mtx = []
        logging.info('loading e att res from [%s]', self.in_name)
        for line in open(self.in_name):
            h = json.loads(line)
            e_att = h['meta']['e_att_score']
            l_e_att_mtx.append(e_att)
        logging.info('att mtx of [%d] q loaded', len(l_e_att_mtx))
        return l_e_att_mtx

    def _normalize_e_att_weights(self, l_e_att_mtx):
        l_res = []
        for e_att_mtx in l_e_att_mtx:
            e_normalized_mtx = []
            for l_sf_e_score in e_att_mtx:
                z = sum([item[1] for item in l_sf_e_score])
                if z:
                    z = float(z)
                    l_sf_e_score = [(item[0], item[1] / z) for item in l_sf_e_score]
                e_normalized_mtx.append(l_sf_e_score)
            l_res.append(e_normalized_mtx)
        logging.info('normalized')
        return l_res

    def ana(self):
        """
        print various stats to out name
        :return:
        """
        logging.info('start ana e attention scores')
        out = open(self.out_name, 'w')
        print >> out, 'cmns e rank distribution:'
        print >> out, json.dumps(self.top_e_rank_dist())
        print >> out, 'cmns e weight frac:'
        print >> out, '%f' % self.top_e_weight_frac()
        print >> out, 'e attention score normalized bins 11 bins in 0-1'
        l_bins = self.att_dist()
        print >> out, json.dumps(zip(range(len(l_bins)), l_bins))
        out.close()
        logging.info('analysis finished')





    def top_e_rank_dist(self):
        """
        the count of top e's rank in each sf's first several e's
        :return:
        """
        l_top_e_rank = []
        for e_att_mtx in self.l_e_att_mtx:
            for l_e_score in e_att_mtx:
                e_id = l_e_score[0][0]

                l_sorted = sorted(l_e_score, key=lambda item: -item[1])
                p = [item[0] for item in l_sorted].index(e_id)
                l_top_e_rank.append(p + 1)
        bins = np.bincount(l_top_e_rank)
        return bins.tolist()

    def top_e_weight_frac(self):
        """
        frac of the first 1's weight
            which is actually a mean of the value cause it is normalized already
        :return:
        """

        l_top_e_weight = []
        for e_att_mtx in self.l_e_att_mtx:
            for l_e_score in e_att_mtx:
                l_top_e_weight.append(l_e_score[0][1])
        frac = sum(l_top_e_weight) / float(len(l_top_e_weight))
        return frac

    def att_dist(self):
        l_scores = []
        for e_att_mtx in self.l_e_att_mtx:
            for l_e_score in e_att_mtx:
                l_scores.extend([int(math.ceil(score * 10)) for e, score in l_e_score])

        bins = np.bincount(l_scores)
        return bins.tolist()


if __name__ == '__main__':
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config
    )
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "ana e att score"
        print "1 para: config"
        EAttAna.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    analysier = EAttAna(config=conf)
    analysier.ana()






