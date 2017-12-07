"""
compute baseline results based on each single feature for events
"""
from traitlets import (
    Unicode,
    Bool,
    Int
)
from traitlets.config import Configurable
from knowledge4ir.salience.utils.data_io import (
    event_feature_io,
    feature_io,
    raw_io
)
from knowledge4ir.salience.utils.evaluation import SalienceEva
from knowledge4ir.utils import add_svm_feature, mutiply_svm_feature

import gzip
from knowledge4ir.utils import (
    body_field,
    abstract_field,
    salience_gold
)
import logging
import json
import numpy as np


class FeatureBasedBaseline(Configurable):
    event_model = Bool(False, help='Run event model').tag(config=True)
    feature_names = Unicode("", help="Comma seperated name of features").tag(
        config=True)
    reverse_feature = Unicode("", help="List the features that should be "
                                       "ranked reversely").tag(config=True)
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

        self.evaluator = SalienceEva(**kwargs)
        self.feature_names_split = self.feature_names.split(",")
        self.feature_dim = len(self.feature_names_split)

        reverse_f = set(self.reverse_feature.split(","))

        # Mask to identify which features should be ranked reversely.
        self.reverse_dim = []
        for i, n in enumerate(self.feature_names_split):
            self.reverse_dim.append(n in reverse_f)

        if self.feature_dim == 0:
            logging.error("You must provide feature names.")
        else:
            logging.info("Number of features to check: %d" % self.feature_dim)

    def eval_per_dim(self, h_packed_data, m_label, reverse_dim):
        features = np.squeeze(h_packed_data['ts_feature'].data.numpy(), axis=0)
        labels = np.squeeze(m_label.data.numpy(), axis=0)

        eval_res = []

        for f_dim in range(features.shape[1]):
            values = features[:, f_dim]
            if reverse_dim[f_dim]:
                values = [0 - v for v in values]
            eval_res.append(self.evaluator.evaluate(values, labels))

        return eval_res

    def process(self):
        open_func = gzip.open if self.corpus_in.endswith("gz") else open
        with open_func(self.corpus_in) as in_f:
            l_h_total_eva = [{} for _ in range(self.feature_dim)]
            p = 0
            for line in in_f:
                if self._filter_empty_line(line):
                    continue

                p += 1
                # Instead of providing batch, we just give one by one.
                h_packed_data, m_label = self.io(
                    [line], self.spot_field, self.in_field, self.salience_gold,
                    None
                )
                res = self.eval_per_dim(h_packed_data, m_label,
                                        self.reverse_dim)

                for dim, h_this_eva in enumerate(res):
                    l_h_total_eva[dim] = add_svm_feature(l_h_total_eva[dim],
                                                         h_this_eva)
                    h_mean_eva = mutiply_svm_feature(l_h_total_eva[dim],
                                                     1.0 / p)

                    if not p % 1000:
                        logging.info('predicted [%d] docs, eva %s for [%s]', p,
                                     json.dumps(h_mean_eva),
                                     self.feature_names_split[dim])

            for dim, h_total_eva in enumerate(l_h_total_eva):
                h_mean_eva = mutiply_svm_feature(h_total_eva, 1.0 / p)
                logging.info('finished predicted [%d] docs, eva %s for [%s]', p,
                             json.dumps(h_mean_eva),
                             self.feature_names_split[dim])

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
