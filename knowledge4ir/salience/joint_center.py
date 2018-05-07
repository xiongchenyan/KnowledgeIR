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
from __future__ import print_function

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

    FeatureConcatKernelCRF,

    MultiEventKernelCRF,

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
    event_only = Bool(False, help='whether to run event model only').tag(
        config=True)
    multi_output = Bool(False, help='whether there are multiple output').tag(
        config=True
    )

    def __init__(self, **kwargs):
        joint_models = {
            'masked_linear_kcrf': MaskKernelCrf,

            'multi_kcrf': MultiEventKernelCRF,

            'concat_linear_kcrf': FeatureConcatKernelCRF,

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

        self.train_losses = {}
        self.batch_count = 0
        self.epoch = 0
        self.data_count = 0

        assert not (self.multi_output and self.event_only)

        if self.multi_output:
            self.output_names = ['entity', 'event']
            logging.info('Using multi-task output.')
        elif self.event_only:
            self.output_names = ['event']
            logging.info('Event only model.')
        else:
            self.output_names = ['entity', 'event']

    def __init_batch_info(self):
        if self.multi_output:
            for t in self.output_names:
                self.train_losses[t] = 0
        self.train_losses['joint'] = 0
        self.batch_count = 0
        self.data_count = 0

    def _setup_io(self, **kwargs):
        self.io_parser = EventDataIO(**kwargs)

    def _init_model(self):
        if self.model_name:
            if not self.event_only:
                self._merge_para()
            self.model = self.h_model[self.model_name](self.para, self.ext_data)
            logging.info('use model [%s]', self.model_name)

    def load_model(self, model_out_name):
        super(JointSalienceModelCenter, self).load_model(model_out_name)
        # Hacking: fix some version incompatible.
        if isinstance(self.model, MultiEventKernelCRF):
            # Old trained model does not have this variable set.
            self.model.arg_voting = self.para.arg_voting

    def train(self, train_in_name, validation_in_name=None,
              model_out_name=None):
        if not model_out_name:
            model_out_name = train_in_name + '.model_%s' % self.model_name
        name, ext = os.path.splitext(model_out_name)
        model_out_name = name + "_" + self.init_time + ext
        super(JointSalienceModelCenter, self).train(train_in_name,
                                                    validation_in_name,
                                                    model_out_name)

    def _batch_train(self, l_line, criterion, optimizer):
        self.batch_count += 1
        self.data_count += len(l_line)

        if self.multi_output:
            h_packed_data, l_m_label = self._data_io(l_line)
            optimizer.zero_grad()
            l_output = self.model(h_packed_data)
            assert len(l_output) == len(l_m_label)

            l_loss = [] * len(l_output)
            for output, m_label, n in zip(l_output, l_m_label,
                                          self.output_names):
                if output is not None:
                    loss = criterion(output, m_label)
                    l_loss.append(loss)
                    self.train_losses[n] += loss.data[0]

            total_loss = sum(l_loss)

            # Joint loss for information only.
            joint_output = torch.cat(l_output, -1)
            joint_label = torch.cat(l_m_label, -1)
            joint_loss = criterion(joint_output, joint_label)

            self.train_losses['joint'] += joint_loss.data[0]

            total_loss.backward()
            optimizer.step()
            assert not math.isnan(total_loss.data[0])
            return total_loss.data[0]
        else:
            return super(JointSalienceModelCenter, self)._batch_train(l_line,
                                                                      criterion,
                                                                      optimizer)

    def _batch_test(self, l_lines):
        if self.multi_output:
            h_packed_data, l_m_label = self._data_io(l_lines)
            l_output = self.model(h_packed_data)

            l_loss = []

            for output, m_label in zip(l_output, l_m_label):
                loss = self.criterion(output, m_label)
                l_loss.append(loss)
            total_loss = sum(l_loss)
            return total_loss.data[0]
        else:
            return super(JointSalienceModelCenter, self)._batch_test(l_lines)

    def _epoch_start(self):
        self.__init_batch_info()

    def _train_info(self):
        if self.multi_output:
            for n, loss in self.train_losses.items():
                logging.info(
                    'Loss [%s], epoch [%d] finished with loss [%f] on [%d] '
                    'batch [%d] doc', n, self.epoch, loss / self.batch_count,
                    self.batch_count, self.data_count)

    def predict(self, test_in_name, label_out_name, debug=False,
                timestamp=True):
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
        self.model.eval()

        name, ext = os.path.splitext(label_out_name)
        if timestamp:
            ent_label_out_name = name + "_entity_" + self.init_time + ext
            evm_label_out_name = name + "_event_" + self.init_time + ext
        else:
            ent_label_out_name = name + "_entity" + ext
            evm_label_out_name = name + "_event" + ext

        ent_out = open(ent_label_out_name, 'w')
        evm_out = open(evm_label_out_name, 'w')

        outs = [ent_out, evm_out]

        logging.info('start predicting for [%s]', test_in_name)
        logging.info('Test output will be at [%s] and [%s]',
                     ent_label_out_name, evm_label_out_name)

        p = 0
        ent_p = 0
        evm_p = 0

        h_total_ent_eva = dict()
        h_total_evm_eva = dict()
        for line in open(test_in_name):
            if self.io_parser.is_empty_line(line):
                continue
            l_h_out = self._per_doc_predict(line)

            if not l_h_out:
                continue

            for h_out, name, out in zip(l_h_out, self.output_names, outs):
                if not h_out:
                    continue

                out.write(json.dumps(h_out) + '\n')

                eva = h_out['eval']
                if name == 'entity':
                    ent_p += 1
                    h_total_ent_eva = add_svm_feature(h_total_ent_eva, eva)
                if name == 'event':
                    evm_p += 1
                    h_total_evm_eva = add_svm_feature(h_total_evm_eva, eva)

            p += 1

            if not p % 1000:
                h_mean_ent_eva = mutiply_svm_feature(h_total_ent_eva,
                                                     1.0 / max(ent_p, 1.0))

                h_mean_evm_eva = mutiply_svm_feature(h_total_evm_eva,
                                                     1.0 / max(evm_p, 1.0))

                logging.info('predicted [%d] docs: [%d] with entities, eva %s;'
                             '[%d] with events, eva %s',
                             p, ent_p, json.dumps(h_mean_ent_eva),
                             evm_p, json.dumps(h_mean_evm_eva),
                             )

        h_mean_ent_eva = mutiply_svm_feature(h_total_ent_eva,
                                             1.0 / max(ent_p, 1.0))
        h_mean_evm_eva = mutiply_svm_feature(h_total_evm_eva,
                                             1.0 / max(evm_p, 1.0))

        l_mean_ent_eva = sorted(h_mean_ent_eva.items(),
                                key=lambda item: item[0])
        l_mean_evm_eva = sorted(h_mean_evm_eva.items(),
                                key=lambda item: item[0])

        logging.info('finished predicted [%d] docs, [%d] with entities, eva %s'
                     '[%d] with events, eva %s', p, ent_p,
                     json.dumps(l_mean_ent_eva), evm_p,
                     json.dumps(l_mean_evm_eva))

        self.tab_scores(h_mean_ent_eva, h_mean_evm_eva)

        json.dump(
            l_mean_ent_eva,
            open(ent_label_out_name + '.eval', 'w'),
            indent=1
        )

        json.dump(
            l_mean_evm_eva,
            open(evm_label_out_name + '.eval', 'w'),
            indent=1
        )

        ent_out.close()
        evm_out.close()
        return

    @staticmethod
    def tab_scores(h_e_mean_eva, h_evm_mean_eva):
        logging.info("Results to copy to Excel:")

        line1 = ["p@01", "p@05", "p@10", "p@20", "auc"]
        line2 = ["r@01", "r@05", "r@10", "r@20"]

        def get(h, k):
            return "%.4f" % h[k] if k in h else "-"

        l1_evm_scores = [get(h_evm_mean_eva, k) for k in line1]
        l1_ent_scores = [get(h_e_mean_eva, k) for k in line1]
        l1_all_scores = ['-' for _ in line1]

        l2_evm_scores = [get(h_evm_mean_eva, k) for k in line2]
        l2_ent_scores = [get(h_e_mean_eva, k) for k in line2]
        l2_all_scores = ['-' for _ in line2]

        print("\t-\t".join(l1_evm_scores) + "\t-\t-\t" +
              "\t".join(l1_all_scores) + "\t-\t" +
              "\t".join(l1_ent_scores))

        print("\t-\t".join(l2_evm_scores) + "\t-\t-\t-\t-\t" +
              "\t".join(l2_all_scores) + "\t-\t-\t" +
              "\t".join(l2_ent_scores))

    def _multi_output(self, line, key_name, docno):
        h_packed_data, l_v_label = self._data_io([line])

        l_h_out = [dict() for _ in range(len(self.output_names))]

        l_output = []
        for output in self.model(h_packed_data):
            if output is None:
                l_output.append([])
            else:
                l_output.append(output.cpu()[0])

        for i, name in enumerate(self.output_names):
            output = l_output[i]
            v_label = l_v_label[i]

            if v_label is None or not v_label[0].size():
                continue

            pre_label = output.data.sign().type(torch.LongTensor)
            l_score = output.data.numpy().tolist()
            l_h_out[i][key_name] = docno
            l_label = v_label[0].cpu().data.view_as(
                pre_label).numpy().tolist()

            if name == 'entity':
                mtx_e = h_packed_data['mtx_e']
                l_e = mtx_e[0].cpu().data.numpy().tolist()
            else:
                mtx_e = h_packed_data['mtx_evm']
                l_e = mtx_e[0].cpu().data.numpy().tolist()
                l_e = [e - self.entity_range for e in l_e]

            # Add output.
            l_h_out[i][self.io_parser.content_field] = {
                'predict': zip(l_e, l_score)
            }

            if l_label:
                h_this_eva = self.evaluator.evaluate(l_score, l_label)
            else:
                h_this_eva = {}

            l_h_out[i]['eval'] = h_this_eva
        return l_h_out

    def _single_output(self, line, key_name, docno):
        l_h_out = [dict()]

        h_packed_data, v_label = self._data_io([line])

        if v_label is None or not v_label[0].size():
            return None
        mtx_e = h_packed_data['mtx_e']
        l_e = mtx_e[0].cpu().data.numpy().tolist()
        output = self.model(h_packed_data).cpu()[0]
        pre_label = output.data.sign().type(torch.LongTensor)
        l_score = output.data.numpy().tolist()

        l_label = v_label[0].cpu().data.view_as(pre_label).numpy().tolist()
        h_this_eva = self.evaluator.evaluate(l_score, l_label)

        l_h_out[0][key_name] = docno
        l_h_out[0]['eval'] = h_this_eva
        l_h_out[0][self.io_parser.content_field] = {
            'predict': zip(l_e, l_score)
        }
        return l_h_out

    def _merged_output(self, line, key_name, docno):
        l_h_out = [{}] * len(self.output_names)

        h_combined = {key_name: docno}
        l_h_out[0][key_name] = docno
        l_h_out[1][key_name] = docno

        h_packed_data, v_label = self._data_io([line])

        if v_label is None or not v_label[0].size():
            return None

        mtx_e = h_packed_data['mtx_e']
        l_combined = mtx_e[0].cpu().data.numpy().tolist()

        l_e = [e for e in l_combined if e < self.entity_range]
        l_evm = [e for e in l_combined if e >= self.entity_range]

        output = self.model(h_packed_data).cpu()[0]

        pre_label = output.data.sign().type(torch.LongTensor)
        l_score = output.data.numpy().tolist()

        l_label = v_label[0].cpu().data.view_as(pre_label).numpy().tolist()

        num_entities = sum(
            [1 if e < self.entity_range else 0 for e in l_e]
        )

        l_label_e = l_label[:num_entities]
        l_score_e = l_score[:num_entities]

        l_label_evm = l_label[num_entities:]
        l_score_evm = l_score[num_entities:]
        l_evm_origin = [e - self.entity_range for e in l_evm]

        l_h_out[0][self.io_parser.content_field] = {
            'predict': zip(l_e, l_score_e)
        }

        l_h_out[1][self.io_parser.content_field] = {
            'predict': zip(l_evm_origin, l_score_evm)
        }

        h_combined[self.io_parser.content_field] = {
            'predict': zip(l_combined, l_score)
        }

        h_this_eva = self.evaluator.evaluate(l_score, l_label)
        h_combined['eval'] = h_this_eva

        if l_label_e:
            h_entity_eva = self.evaluator.evaluate(l_score_e, l_label_e)
        else:
            h_entity_eva = {}

        if l_label_evm:
            h_evm_eva = self.evaluator.evaluate(l_score_evm, l_label_evm)
        else:
            h_evm_eva = {}

        l_h_out[0]['eval'] = h_entity_eva
        l_h_out[1]['eval'] = h_evm_eva

        return l_h_out

    def _per_doc_predict(self, line):
        h_info = json.loads(line)
        key_name = 'docno'
        if key_name not in h_info:
            key_name = 'qid'
            assert key_name in h_info
        docno = h_info[key_name]

        if self.multi_output:
            return self._multi_output(line, key_name, docno)
        elif self.event_only:
            return self._single_output(line, key_name, docno)
        else:
            return self._merged_output(line, key_name, docno)

    def _data_io(self, l_line):
        return self.model.data_io(l_line, self.io_parser)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
        load_command_line_config,
    )

    from traitlets.config.loader import KeyValueConfigLoader


    class Main(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        model_in = Unicode(help='model to read from').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        skip_train = Bool(False, help='directly test').tag(config=True)
        debug = Bool(False, help='Debug mode').tag(config=True)
        time_stamp = Bool(True, help='Add time stamp to output.').tag(
            config=True)


    if len(sys.argv) < 2:
        print("Please provide setting file: [this script] [setting]")
        JointSalienceModelCenter.class_print_help()
        Main.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    cl_conf = load_command_line_config(sys.argv[2:])
    conf.merge(cl_conf)

    para = Main(config=conf)
    assert para.test_in
    assert para.test_out

    set_basic_log(logging.getLevelName(para.log_level))

    model = JointSalienceModelCenter(config=conf)

    model_loaded = False
    if para.skip_train:
        logging.info('Trying to load existing model.')
        if os.path.exists(para.model_in):
            model.load_model(para.model_in)
            model_loaded = True
        else:
            logging.info("Cannot find model [%s], "
                         "please set exact path." % para.model_in)

    if not model_loaded:
        logging.info('Start to run training.')
        model.train(para.train_in, para.valid_in, para.model_out)

    model.predict(para.test_in, para.test_out, para.debug, para.time_stamp)
