"""
compute baseline results based on each single feature for events
"""
from traitlets import (
    Unicode,
    Bool
)
from traitlets.config import Configurable
from knowledge4ir.salience.utils.data_io import event_feature_io, feature_io, \
    raw_io
import gzip
from knowledge4ir.utils import (
    body_field,
    abstract_field,
    salience_gold
)
import logging
import json


class FeatureBasedBaseline(Configurable):
    event_model = Bool(False, help='Run event model').tag(config=True)
    corpus_in = Unicode(help='input').tag(config=True)
    test_out = Unicode(help='output').tag(config=True)

    in_field = Unicode(body_field)
    salience_field = Unicode(abstract_field)
    spot_field = Unicode('spot')
    # A specific field is reserved to mark the salience answer.
    salience_gold = Unicode(salience_gold)

    io = feature_io

    def __init__(self, **kwargs):
        super(FeatureBasedBaseline, self).__init__(**kwargs)
        if self.event_model:
            self.spot_field = 'event'
        if self.event_model:
            self.io = event_feature_io

    def rank_on_dim(self, h_packed_data, m_label):
        features = h_packed_data['ts_feature'].data
        labels = m_label.data
        print features
        print labels

    def process(self):
        open_func = gzip.open if self.corpus_in.endswith("gz") else open
        with open_func(self.corpus_in) as in_f:
            for line in in_f:
                if self._filter_empty_line(line):
                    continue

                # Instead of providing batch, we just give one by one.
                h_packed_data, m_label = self.io(
                    [line], self.spot_field, self.in_field, self.salience_gold,
                    None
                )
                self.rank_on_dim(h_packed_data, m_label)
                import sys
                sys.stdin.readline()

    def _filter_empty_line(self, line):
        h = json.loads(line)
        if self.io == raw_io:
            l_e = h[self.spot_field].get(self.in_field, [])
        elif self.io == event_feature_io:
            l_e = h[self.spot_field].get(self.in_field, {}).get('salience')
        else:
            l_e = h[self.spot_field].get(self.in_field, {}).get('entities')
        return not l_e


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log
    )

    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "hashing corpus, 1 para, config:"
        FeatureBasedBaseline.class_print_help()
        sys.exit(-1)

    use_cuda = False

    runner = FeatureBasedBaseline(config=load_py_config(sys.argv[1]))
    runner.process()
