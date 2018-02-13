"""
compute baseline results based on each single feature for events
"""
from traitlets import (
    Unicode,
    Bool,
    Int
)
from traitlets.config import Configurable
from knowledge4ir.salience.utils.joint_data_io import (
    EventDataIO
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
import torch

use_cuda = torch.cuda.is_available()


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

    def __init__(self, **kwargs):
        super(FeatureBasedBaseline, self).__init__(**kwargs)
        if self.event_model:
            self.spot_field = 'event'

        self.io = EventDataIO(**kwargs)

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

    def eval_per_dim(self, h_packed_data, m_label, reverse_dim, key_name,
                     docno):
        if use_cuda:
            feature_data = h_packed_data['ts_feature'].data.cpu()
            label_data = m_label.data.cpu()
        else:
            feature_data = h_packed_data['ts_feature'].data
            label_data = m_label.data

        features = np.squeeze(feature_data.numpy(), axis=0)
        labels = np.squeeze(label_data.numpy(), axis=0).tolist()

        num_features = features.shape[1]
        l_h_out = [dict() for _ in range(num_features)]

        for f_dim in range(num_features):
            values = features[:, f_dim].tolist()
            l_h_out[f_dim][key_name] = docno
            mtx_e = h_packed_data['mtx_e']

            if use_cuda:
                l_e = mtx_e[0].cpu().data.numpy().tolist()
            else:
                l_e = mtx_e[0].data.numpy().tolist()

            if reverse_dim[f_dim]:
                values = [0 - v for v in values]
            l_h_out[f_dim][body_field] = {'predict': zip(l_e, values)}
            l_h_out[f_dim]['eval'] = self.evaluator.evaluate(values, labels)

        return l_h_out

    def process(self):
        open_func = gzip.open if self.corpus_in.endswith("gz") else open

        outs = []
        for name in self.feature_names_split:
            out_path = self.test_out + "_" + name.replace(" ", "_") + '.json'
            outs.append(open(out_path, 'w'))
            logging.info("Feature output will be stored at [%s]" % out_path)

        with open_func(self.corpus_in) as in_f:
            l_h_total_eva = [{} for _ in range(self.feature_dim)]
            p = 0
            for line in in_f:
                if self.io.is_empty_line(line):
                    continue

                # Instead of providing batch, we just give one by one.
                h_packed_data, m_label = self.io.parse_data([line])

                h_info = json.loads(line)
                key_name = 'docno'
                docno = h_info[key_name]

                p += 1
                l_h_out = self.eval_per_dim(h_packed_data, m_label,
                                            self.reverse_dim, key_name, docno)

                for (dim, h_out), out in zip(enumerate(l_h_out), outs):
                    h_this_eva = h_out['eval']
                    l_h_total_eva[dim] = add_svm_feature(l_h_total_eva[dim],
                                                         h_this_eva)
                    h_mean_eva = mutiply_svm_feature(l_h_total_eva[dim],
                                                     1.0 / p)

                    print >> out, json.dumps(h_out)

                    if not p % 1000:
                        logging.info('predicted [%d] docs, eva %s for [%s]', p,
                                     json.dumps(h_mean_eva),
                                     self.feature_names_split[dim])

            for dim, h_total_eva in enumerate(l_h_total_eva):
                h_mean_eva = mutiply_svm_feature(h_total_eva, 1.0 / p)
                logging.info('finished predicted [%d] docs, eva %s for [%s]', p,
                             json.dumps(h_mean_eva),
                             self.feature_names_split[dim])

        for (dim, h_total_eva), name in zip(enumerate(l_h_total_eva),
                                            self.feature_names_split):
            h_mean_eva = mutiply_svm_feature(h_total_eva, 1.0 / p)
            l_mean_eva = sorted(h_mean_eva.items(),
                                key=lambda item: item[0])

            logging.info('finished predicted [%d] docs, eva %s', p,
                         json.dumps(l_mean_eva))

            with open(self.test_out + "_" + name.replace(" ", "_") + '.eval',
                      'w') as o:
                json.dump(l_mean_eva, o, indent=1)

        for out in outs:
            out.close()


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

    runner = FeatureBasedBaseline(config=load_py_config(sys.argv[1]))
    runner.process()
