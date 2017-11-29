"""
compute baseline results based on each single feature for events
"""
from traitlets import (
    Unicode,
    Int,
    Float,
    List,
    Bool
)
from traitlets.config import Configurable
from knowledge4ir.salience.utils.data_io import event_feature_io
import gzip
from knowledge4ir.utils import (
    body_field,
    abstract_field,
    salience_gold
)


class FeatureBasedBaseline(Configurable):
    max_e_per_doc = Int(200, help='max e per doc')
    event_model = Bool(False, help='Run event model').tag(config=True)
    corpus_in = Unicode(help='input').tag(config=True)

    in_field = Unicode(body_field)
    salience_field = Unicode(abstract_field)
    spot_field = Unicode('spot')
    # A specific field is reserved to mark the salience answer.
    salience_gold = Unicode(salience_gold)

    def process(self):
        open_func = gzip.open if self.corpus_in.endswith("gz") else open
        with open_func(self.corpus_in) as in_f:
            for line in in_f:
                # Instead of providing batch, we just give one by one.
                h_packed_data, m_label = event_feature_io(
                    [line], self.spot_field, self.in_field, self.salience_gold,
                    self.max_e_per_doc
                )
                print(h_packed_data)
                print(m_label)
