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

from knowledge4ir.salience.translation_model import GraphTranslation
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    Float,
)
import numpy as np
import json
import logging
from knowledge4ir.utils import (
    body_field,
    abstract_field,
    term2lm,
)
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()


class SalienceModelCenter(Configurable):
    learning_rate = Float(1e-3, help='learning rate').tag(config=True)
    pre_trained_emb_in = Unicode(help='pre-trained embedding').tag(config=True)
    model_name = Unicode(help="model name: trans").tag(config=True)
    random_walk_step = Int(1, help='random walk step').tag(config=True)  # need to be a config para
    nb_epochs = Int(2, help='nb of epochs').tag(config=True)

    max_e_per_doc = Int(1000, help='max e per doc')
    h_model = {
        "trans": GraphTranslation,
    }
    in_field = Unicode(body_field)
    salience_field = Unicode(abstract_field)
    spot_field = Unicode('spot')

    def __init__(self, **kwargs):
        super(SalienceModelCenter, self).__init__(**kwargs)
        self.pre_emb = None
        if self.pre_trained_emb_in:
            logging.info('loading pre trained embedding [%s]', self.pre_trained_emb_in)
            self.pre_emb = np.load(open(self.pre_trained_emb_in))
            logging.info('loaded with shape %s', json.dumps(self.pre_emb.shape))
        self.model = None
        self._init_model()

    def _init_model(self):
        if self.model_name:
            self.model = self.h_model[self.model_name](self.random_walk_step,
                                                       self.pre_emb.shape[0],
                                                       self.pre_emb.shape[1],
                                                       self.pre_emb,
                                                       )

    def train(self, train_in_name):
        """
        train using the given data
        will use each doc as the mini-batch for now
        :param train_in_name:
        :return: keep the model
        """
        logging.info('training with data in [%s]', train_in_name)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        l_epoch_loss = []
        for epoch in xrange(self.nb_epochs):
            p = 0
            total_loss = 0
            logging.info('start epoch [%d]', epoch)
            for line in open(train_in_name):
                v_e, v_w, v_label = self._data_io(line)
                if (not v_e.size()) | (not v_label.size()):
                    continue
                optimizer.zero_grad()
                output = self.model(v_e, v_w)
                # print "label"
                # print v_label
                # print "output"
                # print output
                loss = criterion(output, v_label)
                loss.backward()
                optimizer.step()
                total_loss += loss.data[0]
                logging.debug('[%d] data [%f] loss', p, loss.data[0])
                p += 1
                if not p % 100:
                    logging.info('data [%d], average loss [%f]', p, total_loss / p)
            logging.info('epoch [%d] finished with loss [%f] on [%d] data',
                         epoch, total_loss / p, p)
            l_epoch_loss.append(total_loss / p)
        logging.info('[%d] epoch done with loss %s', self.nb_epochs, json.dumps(l_epoch_loss))
        return

    def predict(self, test_in_name, label_out_name):
        """
        predict the data in test_in,
        dump predict labels in label_out_name
        :param test_in_name:
        :param label_out_name:
        :return:
        """

        out = open(label_out_name, 'w')
        logging.info('start predicting for [%s]', test_in_name)
        total_accuracy = 0
        p = 0
        for line in open(test_in_name):
            docno = json.loads(line)['docno']
            v_e, v_w, v_label = self._data_io(line)
            output = self.model(v_e, v_w)
            pre_label = output.data.max(-1, keepdim=True)[1]
            h_out = dict()
            h_out['docno'] = docno
            l_e = v_e.cpu().data.numpy().tolist()
            l_res = pre_label.cpu().data.numpy().tolist()
            h_out['predict'] = zip(l_e, l_res)
            print >> out, json.dumps(h_out)
            this_acc = output.cpu() == v_label.cpu()
            correct = pre_label.eq(v_label.data.view_as(pre_label)).cpu().sum()
            this_acc = np.mean(correct / float(len(l_e)))
            total_accuracy += this_acc
            p += 1
            if not p % 100:
                logging.info('predicted [%d] docs, accuracy [%f]', p, total_accuracy / p)
        logging.info('finished predicting [%d] docs, accuracy [%f]', p, total_accuracy / p)
        out.close()
        return

    def _data_io(self, line):
        """
        convert data to the input for the model
        :param line: the json formatted data
        :return: v_e, v_w, v_label
        v_e: entities in the doc
        v_w: initial weight, TF
        v_label: 1 or -1, salience or not, if label not given, will be 0
        """

        h = json.loads(line)
        l_e = h[self.spot_field].get(self.in_field, [])
        s_salient_e = set(h[self.spot_field].get(self.salience_field, []))
        h_e_tf = term2lm(l_e)
        l_e_tf = sorted(h_e_tf.items(), key=lambda item: -item[1])[:self.max_e_per_doc]
        l_e = [item[0] for item in l_e_tf]
        z = float(sum([item[1] for item in l_e_tf]))
        l_w = [item[1] / z for item in l_e_tf]
        l_label = [1 if e in s_salient_e else 0 for e in l_e]
        v_e = Variable(torch.LongTensor(l_e)).cuda() if use_cuda else Variable(torch.LongTensor(l_e))
        v_w = Variable(torch.FloatTensor(l_w)).cuda() if use_cuda else Variable(torch.FloatTensor(l_w))
        v_label = Variable(torch.LongTensor(l_label)).cuda() if use_cuda else Variable(torch.FloatTensor(l_label))

        return v_e, v_w, v_label


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    set_basic_log(logging.INFO)

    class Main(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)

    if 2 != len(sys.argv):
        print "unit test model train test"
        print "1 para, config"
        SalienceModelCenter.class_print_help()
        Main.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    model = SalienceModelCenter(config=conf)
    para = Main(config=conf)
    model.train(para.train_in)
    model.predict(para.test_in, para.test_out)