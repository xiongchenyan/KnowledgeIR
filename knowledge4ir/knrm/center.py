"""
KNRM center
Implement the API's defined in model.base.ModelBase
keras model constructed via knrm.model
data i/o implemented in knrm.data_reader
hyper-parameter maintained via model.hyper_parameter
"""


from knowledge4ir.model.base import ModelBase
from knowledge4ir.model.hyper_para import HyperParameter
from knowledge4ir.knrm.model import KNRM, AttKNRM
from knowledge4ir.knrm.data_io import (
    pairwise_reader,
    pointwise_reader,
    load_data,
)
import os
import logging
from traitlets import (
    Unicode,
)
from knowledge4ir.utils import (
    load_json_info,
    load_trec_labels_dict,
    load_trec_ranking_with_score,
    dump_trec_out_from_ranking_score,
)
import numpy as np
from keras.callbacks import EarlyStopping


class KNRMCenter(ModelBase):
    model_name = Unicode('KNRM', help='choose from KNRM and AttKNRM').tag(config=True)
    qrel_in = Unicode(help='qrel').tag(config=True)
    q_info_in = Unicode(help='q info tensor, only needed when using raw io').tag(config=True)
    doc_info_in = Unicode(help='doc info tensor, only needed when using raw io').tag(config=True)
    io_format = Unicode(
        'raw',
        help='raw json or npy matrix in a folder, if npy then no meta data is needed,\
         data organized as defined in the model\'s s_target_inputs'
        ).tag(config=True)
    embedding_npy_in = Unicode(
        help='np saved embedding with id aligned, only needed for KNRM'
                               ).tag(config=True)
    h_model = {'KNRM': KNRM, 'AttKNRM': AttKNRM}
    
    def __init__(self, **kwargs):
        super(KNRMCenter, self).__init__(**kwargs)
        self.k_nrm = self.h_model[self.model_name](**kwargs)
        self.hyper_para = HyperParameter(**kwargs)
        if self.embedding_npy_in:
            logging.info('loading embedding for model [%s]', self.model_name)
            emb_mtx = np.load(self.embedding_npy_in)
            self.k_nrm.set_embedding(emb_mtx)
        else:
            logging.info('model [%s] not using embedding', self.model_name)
        self.ranker, self.learner = self.k_nrm.build()
        logging.info('built ranking model:')
        self.ranker.summary()
        logging.info('pairwise training model:')
        self.learner.summary()
        if self.io_format == 'raw':
            self.h_q_info = load_json_info(self.q_info_in, 'qid')
            self.h_doc_info = load_json_info(self.doc_info_in, 'docno')
            self.h_qrel = load_trec_labels_dict(self.qrel_in)

    @classmethod
    def class_print_help(cls, inst=None):
        super(KNRMCenter, cls).class_print_help(inst)
        KNRM.class_print_help(inst)
        AttKNRM.class_print_help(inst)
        HyperParameter.class_print_help(inst)

    def train(self, x, y, hyper_para=None):
        """
        pairwise training
        :param x: the prepared paired input X, should be aligned with _build_model
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
        self.learner.compile(
            hyper_para.opt,
            hyper_para.loss,
        )

        logging.info('start training with [%d] data with batch [%d]', y.shape[0], batch_size)

        res = self.learner.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=hyper_para.nb_epoch,
            callbacks=[EarlyStopping(monitor='loss',
                                     patience=hyper_para.early_stopping_patient
                                     )],
        )
        logging.info('model training finished')
        return res.history['loss'][-1]

    def predict(self, x):
        y = self.ranker.predict(x)
        return y.reshape(-1)

    def train_data_reader(self, in_name, s_target_qid=None):
        if self.io_format == 'raw':
            l_q_rank = load_trec_ranking_with_score(in_name)
            x, y = pairwise_reader(l_q_rank, self.h_qrel, self.h_q_info, self.doc_info_in, s_target_qid)
        else:
            x, y = load_data(os.path.join(in_name, 'pairwise'),
                             self.k_nrm.s_target_inputs, s_target_qid)
        return x, y

    def test_data_reader(self, in_name, s_target_qid=None):
        if self.io_format == 'raw':
            l_q_rank = load_trec_ranking_with_score(in_name)
            x, y = pointwise_reader(l_q_rank, self.h_qrel, self.h_q_info, self.doc_info_in, s_target_qid)
        else:
            x, y = load_data(os.path.join(in_name, 'pointwise'),
                             self.k_nrm.s_target_inputs,
                             s_target_qid)
        return x, y

    def generate_ranking(self, x, out_name):
        """
        the model must be trained
        :param x:
        :param out_name: the place to put the ranking score
        :return:
        """
        y = self.predict(x)
        l_score = y.tolist()
        l_qid = x['qid'].tolist()
        l_docno = x['docno'].tolist()
        dump_trec_out_from_ranking_score(l_qid, l_docno, l_score, out_name, self.model_name)
        logging.info('ranking results dumped to [%s]', out_name)
        return


if __name__ == '__main__':
    """
    show config
    """
    KNRMCenter.class_print_help()