"""
performance at different document length
"""

import json
import logging

from traitlets import (
    Unicode,
    Int,
    List,
)
from traitlets.config import Configurable

from knowledge4ir.result_collect.analysis.base import bin_score
from knowledge4ir.utils import (
    body_field,
)


class EvaVsStat(Configurable):
    target_metric = Unicode('p@05').tag(config=True)
    l_target_stat = List(Unicode, default_value=['doc_len']).tag(config=True)
    content_field = Unicode(body_field).tag(config=True)
    nb_bin = Int(10).tag(config=True)

    def _get_stats(self, h_doc_info):
        h_stat = dict()
        h_stat.update(self._get_doc_len(h_doc_info))
        return h_stat

    def _get_doc_len(self, h_doc_info):
        doc_len = len(h_doc_info.get(self.content_field, []))
        h_stat = {
            'doc_len': doc_len
        }
        return h_stat

    def process(self, in_name, out_name):
        logging.info('compare eva res vs %s', json.dumps(self.l_target_stat))

        l_h_stat = []
        l_score = []
        for p, line in enumerate(open(in_name)):
            if not p % 1000:
                logging.info('processed [%d] lines', p)
            h_doc = json.loads(line)
            h_stat = self._get_stats(h_doc)
            l_h_stat.append(h_stat)
            score = h_doc.get('eval', {}).get(self.target_metric, 0)
            l_score.append(score)
        logging.info('all results loaded, start binning')
        h_stat_bin = dict()
        for stat in self.l_target_stat:
            logging.info('binning [%s]', stat)
            l_stat = [h_stat[stat] for h_stat in l_h_stat]
            l_bin_res, l_bin_range = bin_score(l_stat, list(l_score), self.nb_bin)
            h_stat_bin[stat] = l_bin_res
            h_stat_bin[stat + '_range'] = l_bin_range
            logging.info('[%s] bin %s', stat, json.dumps(l_bin_res))

        json.dump(h_stat_bin, open(out_name, 'w'), indent=1)
        logging.info('finished')


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import load_py_config, set_basic_log
    set_basic_log()
    if 4 > len(sys.argv):
        print "3 para, config + predicted results to analysis + out name"
        EvaVsStat.class_print_help()
        sys.exit(-1)

    ana = EvaVsStat(config=load_py_config(sys.argv[1]))
    ana.process(*sys.argv[2:])



