"""
Baselines using external tools.
"""
import json

from traitlets import (
    Unicode,
    Bool,
    Int
)

from knowledge4ir.utils import (
    body_field,
    abstract_field,
    salience_gold,
    EVENT_SPOT_FIELD,
)
# Require sumy: https://github.com/miso-belica/sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from traitlets.config import Configurable
from collections import defaultdict
from knowledge4ir.salience.utils.evaluation import SalienceEva
from knowledge4ir.utils import add_svm_feature, mutiply_svm_feature

import pickle
import logging
import gzip


class SummarizationBaseline(Configurable):
    # A specific field is reserved to mark the salience answer.
    salience_gold = Unicode(salience_gold)

    corpus_in = Unicode(help='input in text version').tag(config=True)
    test_out = Unicode(help='output').tag(config=True)

    event_id_pickle_in = Unicode(help='pickle of event id').tag(config=True)

    def __init__(self, **kwargs):
        super(SummarizationBaseline, self).__init__(**kwargs)
        self.spot_field = EVENT_SPOT_FIELD

        self.evaluator = SalienceEva(**kwargs)

        lang = 'english'
        stemmer = Stemmer(lang)
        self.summarizer = Summarizer(stemmer)
        self.summarizer.stop_words = get_stop_words(lang)

        self.h_event_id = pickle.load(open(self.event_id_pickle_in))

    def get_event_head(self, event_info):
        for f in event_info['feature']['sparseFeatureArray']:
            if f.startswith('LexicalHead_'):
                return f.split('_')[1]

    def is_empty(self, data):
        l_s = data[self.spot_field].get(body_field, [])
        return not l_s

    def process(self):
        h_total_eva = {}
        with gzip.open(self.corpus_in) as test_in, \
                open(self.test_out, 'w') as out:
            p = 0

            for line in test_in:
                data = json.loads(line)
                if self.is_empty(data):
                    continue

                p += 1

                word2eid = defaultdict(list)

                labels = []
                l_e = []

                index = 0
                for event in data[self.spot_field][body_field]:
                    word2eid[event['surface']].append(index)
                    labels.append(event['salience'])
                    event_id = self.h_event_id.get(self.get_event_head(event),
                                                   0)
                    l_e.append(event_id)

                text = data[body_field]
                parser = PlaintextParser.from_string(text, Tokenizer('english'))

                predicted = {}

                rank = 1
                for sentence in self.summarizer(parser.document, 10):
                    for word in sentence.words:
                        if word in word2eid:
                            eids = word2eid[word]
                            if word not in predicted:
                                predicted[word] = (eids, rank)
                                rank += 1

                prediction = [0] * len(labels)
                for w, (eids, rank) in predicted.items():
                    for eid in eids:
                        prediction[eid] = 1.0 / rank

                eva = self.evaluator.evaluate(prediction, labels)

                h_out = {
                    'docno': data['docno'],
                    body_field: {
                        'predict': zip(l_e, prediction),
                    },
                    'eval': eva,
                }

                h_total_eva = add_svm_feature(h_total_eva, eva)
                h_mean_eva = mutiply_svm_feature(h_total_eva, 1.0 / p)

                print >> out, json.dumps(h_out)

                if not p % 1000:
                    logging.info('predicted [%d] docs, eva %s', p,
                                 json.dumps(h_mean_eva))


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log
    )

    set_basic_log(logging.INFO)

    runner = SummarizationBaseline(config=load_py_config(sys.argv[1]))
    runner.process()
