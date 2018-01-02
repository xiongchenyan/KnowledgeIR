"""
calculate the stats of the salince datasets

in:
    hashed corpus (train, dev, or test)
output:
    # of doc
    # of words
    # of e
    # of salient e
    word vocab size
    entity vocab size

"""

import json
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
)
from knowledge4ir.utils import (
    body_field,
    paper_abstract_field,
)


class DatasetStat(Configurable):
    content_field = Unicode(body_field).tag(config=True)
    in_name = Unicode().tag(config=True)
    out_name = Unicode().tag(config=True)

    def _count_per_doc(self, h_d_info):
        """

        :param h_d_info:
        :return: a dict of:
            word_cnt:
            e_cnt:
            salient_e_cnt:
            set_word:
            set_e:
        """

        w_cnt = len(h_d_info.get(self.content_field, []))
        s_w = set(h_d_info.get(self.content_field, []))

        h_ana = h_d_info.get('spot', {}).get(self.content_field, {})
        l_e = h_ana.get('entities', [])
        l_label = h_ana.get('salience', [])
        l_feature = h_ana.get('features', [])
        s_e = set(l_e)
        e_cnt = 0
        salience_e_cnt = 0
        if l_label:
            for i in xrange(len(l_label)):
                e_cnt += l_feature[i][0]
                if l_label[i] > 0:
                    salience_e_cnt += l_feature[i][0]

        h_res = {
            'word_cnt': w_cnt,
            'e_cnt': e_cnt,
            'salience_e_cnt': salience_e_cnt,
            'set_word': s_w,
            'set_e': s_e,
            'd_cnt': 0
        }
        return h_res

    def process(self, in_name=None, out_name=None):
        if not in_name:
            in_name, out_name = self.in_name, self.out_name

        h_total_set = {
            'word_cnt': 0,
            'e_cnt': 0,
            'salience_e_cnt': 0,
            'd_cnt': 0
        }
        set_word = set()
        set_e = set()

        for p, line in enumerate(open(in_name)):
            if not p % 1000:
                print "processed [%d] docs" % p

            h_total_set['d_cnt'] += 1
            h_d_info = json.loads(line)
            h_res = self._count_per_doc(h_d_info)
            set_word.update(h_res['set_word'])
            set_e.update(h_res['set_e'])
            for key in h_total_set:
                h_total_set[key] += h_res.get(key, 0)

        h_total_set['word_vocab'] = len(set_word)
        h_total_set['entity_vocab'] = len(set_e)
        nb_d = float(h_total_set['d_cnt'])
        h_total_set['word_cnt'] /= nb_d
        h_total_set['e_cnt'] /= nb_d
        h_total_set['salience_e_cnt'] /= nb_d

        json.dump(h_total_set, open(out_name, 'w'), indent=1)
        print "finished"
        return


if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config
    import sys

    if 2 >= len(sys.argv):
        print "1+ para: config + in + out (the last two can be in config)"
        DatasetStat.class_print_help()
        sys.exit(-1)

    processor = DatasetStat(config=load_py_config(sys.argv[1]))
    in_name, out_name = (None, None) if len(sys.argv) < 4 else (sys.argv[2], sys.argv[3])
    processor.process(in_name, out_name)
    





