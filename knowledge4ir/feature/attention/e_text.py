"""
textual similarity attention features of entity to the query


"""

from knowledge4ir.feature.attention import EntityAttentionFeature
from traitlets import (
    Unicode,
    List
)
from knowledge4ir.feature import (
    TermStat,
)
from knowledge4ir.utils import (
    text2lm,
)


class EntityTextAttentionFeature(EntityAttentionFeature):
    l_model = List(Unicode,
                   default_value=['lm_dir', 'coordinate', 'tf_idf']
                   ).tag(config=True)
    l_entity_fields = List(Unicode, default_value=['name', 'alias', 'desp']).tag(config=True)
    tagger = Unicode('tagme', help='tagger used, as in q info and d info'
                     ).tag(config=True)

    def __init__(self, **kwargs):
        super(EntityTextAttentionFeature, self).__init__(**kwargs)
        self.s_model = set(self.l_model)
        self.h_field_h_df = {}
        self.h_corpus_stat = {}
        self.h_entity_texts = {}

    def set_external_info(self, external_info):
        super(EntityTextAttentionFeature, self).set_external_info(external_info)
        self.h_field_h_df = external_info.h_field_h_df
        self.h_corpus_stat = external_info.h_corpus_stat
        self.h_entity_texts = external_info.h_entity_texts
        return

    def extract(self, h_q_info, l_e):
        l_h_feature = []
        for e in l_e:
            h_feature = {}
            h_feature.update(self._extract_per_e(h_q_info, e))
            l_h_feature.append(h_feature)
        return l_h_feature

    def _extract_per_e(self, h_q_info, e):

        h_feature = {}
        total_df = self.h_corpus_stat['bodyText']['total_df']
        avg_doc_len = self.h_corpus_stat['bodyText']['average_len']
        h_doc_df = self.h_field_h_df['bodyText']
        q_lm = text2lm(h_q_info['query'])

        for e_field in self.l_entity_fields:
            cnt = 0
            e_text = ""
            if e in self.h_entity_texts:
                if e_field in self.h_entity_texts[e]:
                    e_text = self.h_entity_texts[e][e_field]
                    if type(e_text) == list:
                        e_text = ' '.join(e_text)
            e_lm = text2lm(e_text, clean=True)
            cnt += 1
            term_stat = TermStat()
            term_stat.set_from_raw(q_lm, e_lm, h_doc_df, total_df, avg_doc_len)
            l_sim_score = term_stat.mul_scores()
            for sim, score in l_sim_score:
                if sim in self.s_model:
                    h_feature[self.feature_name_pre + e_field.title() + sim.title()] = score

        return h_feature
