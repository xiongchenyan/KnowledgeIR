"""
model I/O, train, and testing center
train:
    hashed nyt data
        three field:
            docno:
            body: l_e
            abstract: l_e
    and train the model
test:
    hashed nyt data
    output the scores for entities in body

hyper-parameters:
    mini-batch size
    learning rate
    vocabulary size
    embedding dim

"""

import json
import logging
import math
import os

import numpy as np
import torch
import datetime
from traitlets import (
    Unicode,
    Int,
    Float,
    List,
    Bool
)
from traitlets.config import Configurable

from knowledge4ir.salience.center import SalienceModelCenter
from knowledge4ir.salience.graph_model import (
    MaskKernelCrf,

    AverageEventKernelCRF,
    AverageArgumentKernelCRF,

    GraphCNNKernelCRF,
    ConcatGraphCNNKernelCRF,
    ResidualGraphCNNKernelCRF,
)
from knowledge4ir.salience.utils.joint_data_io import EventDataIO

from knowledge4ir.utils import (
    add_svm_feature,
    mutiply_svm_feature,
)

use_cuda = torch.cuda.is_available()


class JointSalienceModelCenter(SalienceModelCenter):

    def __init__(self, **kwargs):
        joint_models = {
            'masked_linear_kcrf': MaskKernelCrf,

            'kcrf_event_average': AverageEventKernelCRF,
            'kcrf_args_average': AverageArgumentKernelCRF,

            'kcrf_event_gcnn': GraphCNNKernelCRF,
            'kcrf_event_gcnn_concat': ConcatGraphCNNKernelCRF,
            'kcrf_event_gcnn_residual': ResidualGraphCNNKernelCRF,
        }
        self.h_model.update(joint_models)
        self.init_time = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S")
        super(JointSalienceModelCenter, self).__init__(**kwargs)

        # entity_vocab_size is the combined size used to compute matrix sizes,
        # so it is actually the sum of two vocab sizes.
        self.entity_range = self.para.entity_vocab_size - \
                            self.para.event_vocab_size

    def _setup_io(self, **kwargs):
        self.io_parser = EventDataIO(**kwargs)

    def _init_model(self):
        if self.model_name:
            self._merge_para()
            self.model = self.h_model[self.model_name](self.para, self.ext_data)
            logging.info('use model [%s]', self.model_name)

    def train(self, train_in_name, validation_in_name=None,
              model_out_name=None):
        if not model_out_name:
            model_out_name = train_in_name + '.model_%s' % self.model_name
        name, ext = os.path.splitext(model_out_name)
        model_out_name = name + "_" + self.init_time + ext
        super(JointSalienceModelCenter, self).train(train_in_name,
                                                    validation_in_name,
                                                    model_out_name)

    def predict(self, test_in_name, label_out_name, debug=False):
        """
        predict the data in test_in,
        dump predict labels in label_out_name
        :param test_in_name:
        :param label_out_name:
        :param debug:
        :return:
        """
        res_dir = os.path.dirname(label_out_name)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        self.model.debug_mode(debug)

        name, ext = os.path.splitext(label_out_name)
        label_out_name = name + "_" + self.init_time + ext
        ent_label_out_name = name + "_entity_" + self.init_time + ext
        evm_label_out_name = name + "_event_" + self.init_time + ext

        out = open(label_out_name, 'w')
        ent_out = open(ent_label_out_name, 'w')
        evm_out = open(evm_label_out_name, 'w')

        logging.info('start predicting for [%s]', test_in_name)
        logging.info('Test output will be at [%s], [%s] and [%s]',
                     label_out_name, ent_label_out_name, evm_label_out_name)

        p = 0
        ent_p = 0
        evm_p = 0

        h_total_eva = dict()
        h_total_ent_eva = dict()
        h_total_evm_eva = dict()
        for line in open(test_in_name):
            if self.io_parser.is_empty_line(line):
                continue
            h_out, h_ent_out, h_evm_out, h_this_eva, h_ent_eva, h_evm_eva = \
                self._per_doc_predict(line)

            if not h_out:
                continue

            print >> out, json.dumps(h_out)

            if h_ent_eva:
                ent_p += 1
                print >> ent_out, json.dumps(h_ent_out)

            if h_evm_eva:
                evm_p += 1
                print >> evm_out, json.dumps(h_evm_out)

            h_total_eva = add_svm_feature(h_total_eva, h_this_eva)
            h_total_ent_eva = add_svm_feature(h_total_ent_eva, h_ent_eva)

            h_total_evm_eva = add_svm_feature(h_total_evm_eva, h_evm_eva)

            p += 1

            if not p % 1000:
                h_mean_eva = mutiply_svm_feature(h_total_eva, 1.0 / p)

                h_mean_ent_eva = mutiply_svm_feature(h_total_ent_eva,
                                                     1.0 / max(ent_p, 1.0))

                h_mean_evm_eva = mutiply_svm_feature(h_total_evm_eva,
                                                     1.0 / max(evm_p, 1.0))

                logging.info('predicted [%d] docs, eva %s', p,
                             json.dumps(h_mean_eva))
                logging.info('[%d] with entities, eva %s', ent_p,
                             json.dumps(h_mean_ent_eva))
                logging.info('[%d] with events, eva %s', evm_p,
                             json.dumps(h_mean_evm_eva))

        h_mean_eva = mutiply_svm_feature(h_total_eva, 1.0 / max(p, 1.0))
        h_mean_ent_eva = mutiply_svm_feature(h_total_ent_eva,
                                             1.0 / max(ent_p, 1.0))
        h_mean_evm_eva = mutiply_svm_feature(h_total_evm_eva,
                                             1.0 / max(evm_p, 1.0))

        l_mean_eva = sorted(h_mean_eva.items(), key=lambda item: item[0])
        l_mean_ent_eva = sorted(h_mean_ent_eva.items(),
                                key=lambda item: item[0])
        l_mean_evm_eva = sorted(h_mean_evm_eva.items(),
                                key=lambda item: item[0])

        logging.info('finished predicted [%d] docs, eva %s', p,
                     json.dumps(l_mean_eva))
        logging.info('[%d] with entities, eva %s', ent_p,
                     json.dumps(l_mean_ent_eva))
        logging.info('[%d] with events, eva %s', evm_p,
                     json.dumps(l_mean_evm_eva))

        self.tab_scores(h_mean_eva, h_mean_ent_eva, h_mean_evm_eva)

        json.dump(
            l_mean_eva,
            open(label_out_name + '.eval', 'w'),
            indent=1
        )

        ent_out.close()
        evm_out.close()
        out.close()
        return

    def tab_scores(self, h_all_mean_eva, h_e_mean_eva, h_evm_mean_eva):
        logging.info("Results to copy to Excel:")

        line1 = ["p@01", "p@05", "p@10", "p@20", "auc"]
        line2 = ["r@01", "r@05", "r@10", "r@20"]

        l1_evm_scores = ["%.4f" % h_evm_mean_eva[k] for k in line1]
        l1_ent_scores = ["%.4f" % h_e_mean_eva[k] for k in line1]
        l1_all_scores = ["%.4f" % h_all_mean_eva[k] for k in line1]

        l2_evm_scores = ["%.4f" % h_evm_mean_eva[k] for k in line2]
        l2_ent_scores = ["%.4f" % h_e_mean_eva[k] for k in line2]
        l2_all_scores = ["%.4f" % h_all_mean_eva[k] for k in line2]

        print "\t-\t".join(l1_evm_scores) + "\t-\t-\t" + \
              "\t".join(l1_all_scores) + "\t-\t" + \
              "\t".join(l1_ent_scores)

        print "\t-\t".join(l2_evm_scores) + "\t-\t-\t-\t-\t" + \
              "\t".join(l2_all_scores) + "\t-\t-\t" + \
              "\t".join(l2_ent_scores)

    def _per_doc_predict(self, line):
        h_info = json.loads(line)
        key_name = 'docno'
        if key_name not in h_info:
            key_name = 'qid'
            assert key_name in h_info
        docno = h_info[key_name]
        h_packed_data, v_label = self._data_io([line])

        if not v_label[0].size():
            return None, None
        v_label = v_label[0].cpu()

        mtx_e = h_packed_data['mtx_e']
        l_e = mtx_e[0].cpu().data.numpy().tolist()

        l_evm = []
        if 'mtx_evm' in h_packed_data:
            mtx_evm = h_packed_data['mtx_evm']
            if mtx_evm is not None:
                l_evm = mtx_evm[0].cpu().data.numpy().tolist()

        output = self.model(h_packed_data).cpu()[0]

        pre_label = output.data.sign().type(torch.LongTensor)
        l_score = output.data.numpy().tolist()

        l_e_combined = l_e + l_evm

        h_out = dict()
        h_out[key_name] = docno

        h_e_out = dict()
        h_e_out[key_name] = docno

        h_evm_out = dict()
        h_evm_out[key_name] = docno

        y = v_label.data.view_as(pre_label)

        l_label = y.numpy().tolist()

        num_entities = sum(
            [1 if e < self.entity_range else 0 for e in l_e]
        )

        l_label_e = l_label[:num_entities]
        l_score_e = l_score[:num_entities]

        l_label_evm = l_label[num_entities:]
        l_score_evm = l_score[num_entities:]
        l_evm_origin = [e - self.entity_range for e in l_evm]

        # Add output.
        h_out[self.io_parser.content_field] = {
            'predict': zip(l_e_combined, l_score)}

        h_e_out[self.io_parser.content_field] = {
            'predict': zip(l_e, l_score_e)
        }

        h_evm_out[self.io_parser.content_field] = {
            'predict': zip(l_evm_origin, l_score_evm)
        }

        h_this_eva = self.evaluator.evaluate(l_score, l_label)

        if l_label_e:
            h_entity_eva = self.evaluator.evaluate(l_score_e, l_label_e)
        else:
            h_entity_eva = {}

        if l_label_evm:
            h_evm_eva = self.evaluator.evaluate(l_score_evm, l_label_evm)
        else:
            h_evm_eva = {}

        h_out['eval'] = h_this_eva
        h_evm_out['eval'] = h_evm_eva

        return h_out, h_e_out, h_evm_out, h_this_eva, h_entity_eva, h_evm_eva

    def _data_io(self, l_line):
        return self.model.data_io(l_line, self.io_parser)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )


    class Main(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        skip_train = Bool(False, help='directly test').tag(config=True)
        debug = Bool(False, help='Debug mode').tag(config=True)


    if 2 != len(sys.argv):
        print "unit test model train test"
        print "1 para, config"
        JointSalienceModelCenter.class_print_help()
        Main.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    para = Main(config=conf)

    set_basic_log(logging.getLevelName(para.log_level))

    model = JointSalienceModelCenter(config=conf)

    model_loaded = False
    if para.skip_train:
        print 'Trying to load existing model.'
        if os.path.exists(para.model_out):
            model.load_model(para.model_out)
            model_loaded = True

    if not model_loaded:
        print 'Start to run training.'
        model.train(para.train_in, para.valid_in, para.model_out)

    model.predict(para.test_in, para.test_out, para.debug)
