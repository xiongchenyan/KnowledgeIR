import json
import logging
import pickle

from traitlets import Unicode
from traitlets.config import Configurable

from knowledge4ir.utils import body_field, SPOT_FIELD


class AlignPredicted(Configurable):
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
    corpus_type = Unicode('hashed', help='raw or hashed corpus').tag(config=True)

    def __init__(self, **kwargs):
        super(AlignPredicted, self).__init__(**kwargs)
        h_entity_id = pickle.load(open(self.entity_id_pickle_in))
        self.h_eid_entity = dict(
            [(item[1], item[0]) for item in h_entity_id.items()]
        )
        self.h_align_func = {
            'raw': self._align_predict_doc_to_raw_doc,
            'hashed': self._align_predict_to_hashed_doc,
        }

    def align_predict_to_corpus(self, corpus_in, predict_in, out_name):
        h_key_predicted_info = self._load_predict(predict_in)

        out = open(out_name, 'w')
        for p, line in enumerate(open(corpus_in)):
            if p % 1000:
                logging.info('aligned [%d] lines', p)
            h_info = json.loads(line)
            key = self._get_key(h_info)
            logging.debug('aligning [%s]', key)
            if key not in h_key_predicted_info:
                logging.warn('[%s] predicted res not in [%s]',
                             key, predict_in)
            else:
                h_info = self.h_align_func[self.corpus_type](h_info, h_key_predicted_info[key])
            print >> out, json.dumps(h_info)
        out.close()
        logging.info('aligning [%s] to [%s] finished, res [%s]',
                     predict_in, corpus_in, out_name)

    def _load_predict(self, predict_in):
        l_h_predict = [json.loads(line) for line in open(predict_in)]
        assert l_h_predict
        l_keys = [self._get_key(h) for h in l_h_predict]
        h_key_predicted_info = dict(
            zip(l_keys, l_h_predict)
        )
        return h_key_predicted_info

    def _get_key(self, h_info):
        key = h_info.get('docno')
        if not key:
            key = h_info.get('qid')
        return key

    def _align_predict_doc_to_raw_doc(self, h_info, h_prediction):
        """
        add scores from h_prediction to h_info
        :param h_info:
        :param h_prediction:
        :return:
        """
        key = 'docno'
        if key not in h_info:
            key = 'qid'
        l_eid_salience = h_prediction[self.content_field][self.predict_field]
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

    def _align_predict_to_hashed_doc(self, h_hashed_info, h_prediction):
        """

        :param h_hashed_info:
        :param h_prediction:
        :return:
        """
        h_hashed_info['eval'] = h_prediction['eval']
        for field in h_prediction.keys():
            if field not in h_hashed_info['spot']:
                continue
            l_e_score = h_prediction[field][self.predict_field]
            h_e_score = dict(l_e_score)
            l_e = h_hashed_info['spot'][field]['entities']
            l_score = [h_e_score[e] for e in l_e]
            h_hashed_info['spot'][field][self.predict_field] = l_score
        return h_hashed_info


if __name__ == '__main__':
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
        sys,
    )
    set_basic_log(logging.INFO)

    if 4 > len(sys.argv):
        print "align predicted res with raw or hashed corpu"
        print "3+ para: corpus_in + predicted in + out name + config (opt)"
        AlignPredicted.class_print_help()
        sys.exit(-1)

    if 5 <= len(sys.argv):
        aligner = AlignPredicted(config=load_py_config(sys.argv[4]))
    else:
        aligner = AlignPredicted()

    corpus_in, predicted_in, out_name = sys.argv[1:4]
    aligner.align_predict_to_corpus(corpus_in, predicted_in, out_name)

