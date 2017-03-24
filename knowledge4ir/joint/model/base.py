"""
base functions for ranking model
lambda layers for Kernel-pooling
"""
import json
import logging

import numpy as np
from keras.callbacks import EarlyStopping
from traitlets import (
    Unicode,
    List,
)
from knowledge4ir.joint.resource import JointSemanticResource
from knowledge4ir.model.hyper_para import HyperParameter
from knowledge4ir.utils import (
    dump_trec_out_from_ranking_score,
)
from knowledge4ir.model import ModelBase

sf_ground_name = 'sf_ground'
sf_ground_ref = 'sf_ref'
e_ground_name = 'e_ground'
e_ground_ref = 'e_ref'
e_match_name = 'e_match'
ltr_feature_name = 'letor_f'
l_input_name = [sf_ground_name, e_ground_name, e_match_name, ltr_feature_name]
y_name = 'label'


class JointSemanticModel(ModelBase):
    """
    the base class for all models,
    defines API
    """
    model_name = Unicode('joint_semantics')
    aux_pre = Unicode('aux_')
    l_x_name = List(Unicode, default_value=[]).tag(config=True)
    y_name = Unicode(y_name)

    def __init__(self, **kwargs):
        super(JointSemanticModel, self).__init__(**kwargs)
        self.hyper_para = HyperParameter(**kwargs)
        self.external_resource = JointSemanticResource(**kwargs)

        self.ranking_model = None
        self.training_model = None

    def set_external_resource(self, resource):
        self.external_resource = resource

    def _build_model(self):
        raise NotImplementedError

    def _build_para_layers(self):
        raise NotImplementedError

    def _form_model_from_layers(self, h_para_layers):
        raise NotImplementedError

    def train(self, x, y, hyper_para=None):
        return self.pairwise_train(x, y, hyper_para)

    def train_with_dev(self, x, y, dev_x, dev_y, l_hyper_para):
        self.pairwise_train_with_dev(x, y, dev_x, dev_y, l_hyper_para)

    def train_generator(self, in_name, hyper_para, s_target_qid):
        return self.pairwise_train_generator(in_name, hyper_para, s_target_qid)

    def pairwise_train(self, paired_x, y, hyper_para=None):
        """
        pairwise training
        :param paired_x: the prepared paired input X, should be aligned with _build_model
        :param y: label
        :param hyper_para: if set, then use this one
        :return: trained model
        """
        if not hyper_para:
            hyper_para = self.hyper_para
        logging.info('training with para: %s', hyper_para.pretty_print())
        batch_size = hyper_para.batch_size
        if -1 == batch_size:
            batch_size = y.shape[0]
        self._build_model()
        self.training_model.compile(
            hyper_para.opt,
            hyper_para.loss,
        )

        logging.info('start training with [%d] data with full batch', batch_size)

        self.training_model.fit(
            paired_x,
            y,
            batch_size=batch_size,
            nb_epoch=hyper_para.nb_epoch,
            callbacks=[EarlyStopping(monitor='loss',
                                     patience=hyper_para.early_stopping_patient
                                     )],
        )
        logging.info('model training finished')
        return

    def predict(self, x):
        """
        assume the model is trained
        :param x:
        :return:
        """
        y = self.ranking_model.predict(x)
        return y.reshape(-1)

    def generate_ranking(self, x, out_name):
        """
        the model must be trained
        :param x:
        :param out_name: the place to put the ranking score
        :return:
        """
        y = self.predict(x)
        l_score = y.tolist()

        l_qid = [h['qid'] for h in x['meta']]
        l_docno = [h['docno'] for h in x['meta']]

        dump_trec_out_from_ranking_score(l_qid, l_docno, l_score, out_name, self.model_name)
        logging.info('ranking results dumped to [%s]', out_name)
        self.formulate_intermediate_res(x, out_name + 'intermediate_res')
        return

    def pairwise_train_generator(self, in_name, hyper_para, s_target_qid):
        """
        pairwise training with generator
        :param in_name: the prepared paired input
        :param hyper_para: if set, then use this one
        :param s_target_qid: target qid to use
        :return: trained model
        """
        if not hyper_para:
            hyper_para = self.hyper_para
        logging.info('training with para: %s', hyper_para.pretty_print())
        self._build_model()
        self.training_model.compile(
            hyper_para.opt,
            hyper_para.loss,
        )

        logging.info('start training with [%d] target qid', len(s_target_qid))
        steps = len(s_target_qid)
        self.training_model.fit_generator(
            self.pairwise_data_generator(in_name, s_target_qid),
            samples_per_epoch=steps,
            nb_epoch=hyper_para.nb_epoch,
            callbacks=[EarlyStopping(monitor='loss',
                                     patience=hyper_para.early_stopping_patient
                                     )],
        )
        logging.info('generator model training finished')
        return

    def predict_generator(self, in_name, s_target_qid):
        y = self.ranking_model.predict_generator(
            self.pointwise_data_generator(in_name, s_target_qid),
            val_samples=len(s_target_qid)
        )
        return y

    def generate_ranking_generator(self, in_name, out_name, s_target_qid):
        y = self.predict_generator(in_name, s_target_qid)
        l_score = y.tolist()
        l_qid = []
        l_docno = []
        for line in open(in_name):
            h = json.loads(line)
            qid, docno = h['meta']['qid'], h['meta']['docno']
            if qid not in s_target_qid:
                continue
            l_qid.append(qid)
            l_docno.append(docno)
        dump_trec_out_from_ranking_score(l_qid, l_docno, l_score, out_name, self.model_name)
        logging.info('ranking results dumped to [%s]', out_name)
        self.formulate_intermediate_res_generator(in_name, out_name + 'intermediate_res', s_target_qid)
        return

    def formulate_intermediate_res_generator(self, in_name, out_name, s_target_qid):
        raise NotImplementedError

    def formulate_intermediate_res(self, x, out_name):
        raise NotImplementedError

    def hyper_para_dev(self, paired_train_x, train_y, paired_dev_x, dev_y, l_hyper_para):
        """
        return the best hyper_pra in l_hyper_para, based on performance in dev data
        :param paired_train_x:
        :param train_y:
        :param paired_dev_x:
        :param dev_y:
        :param l_hyper_para:
        :return:
        """
        best_loss = None
        best_para = None
        for hyper_para in l_hyper_para:
            self.pairwise_train(paired_train_x, train_y, hyper_para)
            this_loss = self.training_model.evaluate(paired_dev_x, dev_y)
            if not best_loss:
                best_para = hyper_para
                best_loss = this_loss
            if best_loss > this_loss:
                best_loss = this_loss
                best_para = hyper_para
            logging.info('dev fold this loss [%f] best one [%f]', this_loss, best_loss)

        return best_para

    def pairwise_train_with_dev(self, paired_train_x, train_y, paired_dev_x, dev_y, l_hyper_para):
        logging.info('start pairwise train with development for best para')
        best_para = self.hyper_para_dev(paired_train_x, train_y,
                                        paired_dev_x, dev_y,
                                        l_hyper_para)
        logging.info('best para in dev: %s', best_para.pretty_print())

        paired_x = dict()
        Y = np.concatenate(train_y, dev_y)
        for key, tensor in paired_train_x:
            paired_x[key] = np.concatenate(paired_train_x, paired_dev_x)

        self.pairwise_train(paired_x, Y, best_para)
        logging.info('train with best dev para done')
        return

    def train_data_reader(self, in_name, s_target_qid=None):
        return self.pairwise_data_reader(in_name, s_target_qid)

    def test_data_reader(self, in_name, s_target_qid=None):
        return self.pointwise_data_reader(in_name, s_target_qid)

    def train_data_generator(self, in_name, s_target_qid=None):
        return self.pairwise_data_generator(in_name, s_target_qid)

    def test_data_generator(self, in_name, s_target_qid=None):
        return self.pointwise_data_generator(in_name, s_target_qid)

    def pointwise_data_reader(self, in_name, s_target_qid=None):
        """
        read data in in_name, and pack them into kera model X format
        meta.qid and meta.docno are kept
        :param in_name: the data to read,
            each line is a json dict, with those with key in self.l_x_name is the X's list
            those with self.y_name  is the label
            meta.qid is qid
            meta.docno is docno
        :param s_target_qid: the target qid, if None then keep all
        :return: point_X
        """
        logging.info('start generate point wise data from [%s]', in_name)
        l_data = self._simple_reader(in_name, s_target_qid)
        point_data, point_y = self._pack_pointwise(l_data)
        logging.info('pointwise data read')
        return point_data, point_y

    def pairwise_data_reader(self, in_name, s_target_qid=None):
        """
        read prepared data in in_name, pack them for pairwise training
        will pack doc pairs for pairwise training, the second doc's X is added with self.aux_pre
        Y is determined by the two doc's order
        meta.qid, meta.docno, and meta.aux_docno are kept
        :param in_name: the data to read,
            each line is a json dict, with those with key in self.l_x_name is the X's list
            those with self.y_name  is the label
            meta.qid is qid
            meta.docno is docno
        :param s_target_qid: the target qid to keep, if none then keep all
        :return: paired_x
        """
        logging.info('start generating pairwise data from [%s]', in_name)
        l_data = self._simple_reader(in_name, s_target_qid)

        paired_data, paired_y = self._pack_pairwise(l_data)
        logging.info('paired [%d] pointwise data to [%d] pairs', len(l_data), paired_y.shape[0])
        return paired_data, paired_y

    def pointwise_data_generator(self, in_name, s_target_qid):
        """
        yield point wise data of one query each time
        :param in_name: input data
        :param s_target_qid: target qid to keep
        :return:
        """
        for l_data in self._simple_generator(in_name, s_target_qid):
            point_x, point_y = self._pack_pointwise(l_data)
            yield point_x, point_y

    def pairwise_data_generator(self, in_name, s_target_qid):
        """
        yield pairwise data of one query each time
        :param in_name: training/testing data
        :param s_target_qid: target qid to keep
        :return:
        """
        logging.info('start pairwise generating from [%s]', in_name)
        for l_data in self._simple_generator(in_name, s_target_qid):
            pair_x, pair_y = self._pack_pairwise(l_data)
            logging.info('packed into [%d] pairs', pair_y.shape[0])
            if not pair_y.shape[0]:
                logging.info('skip as no pairwise preference for this q')
                continue
            yield pair_x, pair_y

    def _pack_pointwise(self, l_data):
        point_y = []
        h_key_lists = dict()
        for x_name in self.l_x_name:
            h_key_lists[x_name] = []
        l_meta = []
        for data in l_data:
            h_this_x, score = self._pack_one_data(data)
            for key, value in h_this_x.items():
                h_key_lists[key].append(value)
            point_y.append(score)
            l_meta.append(data['meta'])

        logging.debug('converting lists to arrays')
        point_data = self._packed_list_to_array(h_key_lists)
        point_data['meta'] = l_meta
        point_y = np.array(point_y)
        logging.debug('pointwise data packed')
        return point_data, point_y

    def _pack_pairwise(self, l_data):
        h_key_lists = dict()
        for x_name in self.l_x_name:
            h_key_lists[x_name] = []
            h_key_lists[self.aux_pre + x_name] = []
        l_meta = []
        h_qid_instance_cnt = dict()
        paired_y = []
        for i in xrange(len(l_data)):
            x_anchor, y_anchor = self._pack_one_data(l_data[i])
            meta_anchor = l_data[i]['meta']
            q_anchor = meta_anchor['qid']
            logging.debug('start packing pairs start with %s', json.dumps(meta_anchor))
            for j in xrange(i + 1, len(l_data)):
                meta_aux = l_data[j]['meta']
                q_aux = meta_aux['qid']
                if q_aux != q_anchor:
                    break  # only same queries
                x_aux, y_aux = self._pack_one_data(l_data[j])
                if y_aux == y_anchor:
                    continue  # only preference pairs
                y = 1.0
                if y_anchor < y_aux:
                    y = -1.0
                paired_y.append(y)
                for key, value in x_anchor.items():
                    h_key_lists[key].append(value)
                for key, value in x_aux.items():
                    h_key_lists[self.aux_pre + key].append(value)

                this_meta = dict(meta_anchor)
                this_meta[self.aux_pre + 'docno'] = meta_aux['docno']
                l_meta.append(this_meta)

                if q_anchor not in h_qid_instance_cnt:
                    h_qid_instance_cnt[q_anchor] = 1
                else:
                    h_qid_instance_cnt[q_anchor] += 1

        logging.debug('paired [%d] pointwise data to [%d] pairs', len(l_data), len(l_meta))
        logging.debug('pairs per q %s', json.dumps(h_qid_instance_cnt))
        paired_data = self._packed_list_to_array(h_key_lists)
        paired_data['meta'] = l_meta
        paired_y = np.array(paired_y)
        return paired_data, paired_y

    def _pack_one_data(self, data):
        h_res = {}
        for x_name in self.l_x_name:
            if x_name not in data:
                continue
            h_res[x_name] = data[x_name]
        score = data[self.y_name]
        return h_res, score

    @classmethod
    def _packed_list_to_array(cls, h_key_lists):
        """
        convert the packed lists to np array
        :param h_key_lists:
        :return:
        """
        X = dict()
        for key, l_data in h_key_lists.items():
            X[key] = np.array(l_data)
            logging.info('[%s] shape %s', key, json.dumps(X[key].shape))
        return X

    @classmethod
    def _simple_reader(cls, in_name, s_target_qid=None):
        """
        simply read all data and parse them into given format
        :param in_name:
        :param s_target_qid:
        :return:
        """
        nb_target = -1
        if s_target_qid:
            nb_target = len(s_target_qid)
        logging.info('reading from [%s] with [%d] target qid', in_name, nb_target)
        l_data = []
        cnt = 0
        for line in open(in_name):
            cnt += 1
            h = json.loads(line)
            if s_target_qid is not None:
                if h['meta']['qid'] not in s_target_qid:
                    continue
            l_data.append(h)
        logging.info('total [%d] lines [%d] kept', cnt, len(l_data))
        return l_data

    @classmethod
    def _simple_generator(cls, in_name, s_target_qid):
        """
        infinite loop over in_name's lines
        :param in_name: input file
        :param s_target_qid: qid to take
        :return:
        """
        logging.info('generating from [%s] with [%d] target qid', in_name, len(s_target_qid))

        while True:
            logging.debug('(re)starting from beginning of file')
            with open(in_name) as f:
                l_data = []
                current_qid = None
                for p, line in enumerate(f):
                    h = json.loads(line)
                    qid = h['meta']['qid']
                    if qid not in s_target_qid:
                        logging.debug('qid [%s] skipped', qid)
                        continue
                    if current_qid is None:
                        current_qid = qid
                    if qid != current_qid:
                        logging.debug('generated qid [%s]', qid)
                        yield l_data
                        l_data = []
                        current_qid = qid
                    l_data.append(h)
                if l_data:
                    yield l_data





