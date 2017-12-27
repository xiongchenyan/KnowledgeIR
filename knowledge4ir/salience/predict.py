"""
predict data using given model
"""

from knowledge4ir.salience.center import SalienceModelCenter
import logging
import json
import sys
from traitlets.config import Configurable
from traitlets import (
    Unicode
)
import torch
import pickle
from knowledge4ir.utils import body_field, paper_abstract_field, SPOT_FIELD

class PredictedConverter(Configurable):
    """
    convert predicted results of model to raw doc info
    input:
        predicted json
        raw doc/q info json
    output:
        add salience score for each entity in the doc raw json
    """
    entity_id_pickle_in = Unicode(help='pickle of entity id').tag(config=True)
    content_field = Unicode(body_field, help='content field with salience').tag(config=True)
    predict_field = Unicode('predict', help='the field with prediced e salience').tag(config=True)

    def __init__(self, **kwargs):
        super(PredictedConverter, self).__init__(**kwargs)
        h_entity_id = pickle.load(open(self.entity_id_pickle_in))
        self.h_eid_entity = dict(
            [(item[1], item[0]) for item in h_entity_id.items()]
        )

    def _align_one_doc(self, h_info, h_prediction):
        """
        add scores from h_prediction to h_info
        :param h_info:
        :param h_prediction:
        :return:
        """
        key = 'docno'
        if key not in h_info:
            key = 'qid'
        l_eid_salience = h_prediction[self.content_field]['predict']
        l_entity = [self.h_eid_entity.get(eid, 'na') for eid, __ in l_eid_salience]
        h_entity_salience = dict(
            zip(l_entity, [item[1] for item in l_eid_salience])
        )
        l_eid_features = h_prediction[self.content_field].get('predict_features')
        h_eid_features = {}
        if l_eid_features:
            l_entity = [self.h_eid_entity.get(eid, 'na') for eid, __ in l_eid_features]
            h_eid_features = dict(
                zip(l_entity, [item[1] for item in l_eid_features])
            )

        l_ana = h_info.get(SPOT_FIELD, {}).get(self.content_field, [])
        for p in xrange(len(l_ana)):
            entity = l_ana[p]['entities'][0]['id']
            if entity not in h_entity_salience:
                logging.warn('e [%s] not in [%s]\'s prediction', entity, h_info[key])
            l_ana[p]['entities'][0]['salience'] = h_entity_salience.get(entity, 0)
            if l_eid_features:
                l_ana[p]['entities'][0]['salience_feature'] = h_eid_features.get(entity, [])
        if l_ana:
            h_info[SPOT_FIELD][self.content_field] = l_ana
        return h_info

    def align_predict_to_corpus(self, corpus_in, predict_in, out_name):
        h_key_predicted_info = self._load_predict(predict_in)

        out = open(out_name, 'w')
        for p, line in enumerate(open(corpus_in)):
            if p % 1000:
                logging.info('aligned [%d] lines', p)
            h_info = json.loads(line)
            key = h_info.get('docno')
            if not key:
                key = h_info.get('qid')
            logging.debug('aligning [%s]', key)
            if key not in h_key_predicted_info:
                logging.warn('[%s] predicted res not in [%s]',
                             key, predict_in)
            else:
                h_info = self._align_one_doc(h_info, h_key_predicted_info[key])
            print >> out, json.dumps(h_info)
        out.close()
        logging.info('aligning [%s] to [%s] finished, res [%s]',
                     predict_in, corpus_in, out_name)

    def _load_predict(self, predict_in):
        l_h_predict = [json.loads(line) for line in open(predict_in)]
        assert l_h_predict
        key = 'docno'
        if key not in l_h_predict[0]:
            key = 'qid'
            assert key in l_h_predict[0]
        l_keys = [h[key] for h in l_h_predict]
        h_key_predicted_info = dict(
            zip(l_keys, l_h_predict)
        )
        return h_key_predicted_info


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )


    class Main(Configurable):
        test_in = Unicode(help='test in').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)
        log_level = Unicode('INFO', help='log level').tag(config=True)
        raw_corpus_in = Unicode(help='corpus to align').tag(config=True)
        aligned_corpus_out = Unicode(help='aligned corpus output').tag(config=True)


    if 2 > len(sys.argv):
        print "unit test model train test"
        print "1 para, config with aligning config (optional, set if want to align to raw corpus)"
        SalienceModelCenter.class_print_help()
        Main.class_print_help()
        PredictedConverter.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    para = Main(config=conf)

    set_basic_log(logging.getLevelName(para.log_level))

    model = SalienceModelCenter(config=conf)
    model.load_model(para.model_out)
    model.predict(para.test_in, para.test_out)
    converter = PredictedConverter(config=conf)
    if converter.entity_id_pickle_in:
        logging.info('aligning to [%s]', para.raw_corpus_in)
        converter.align_predict_to_corpus(
            para.raw_corpus_in, para.test_out, para.aligned_corpus_out
        )
        logging.info('alignment finished')

