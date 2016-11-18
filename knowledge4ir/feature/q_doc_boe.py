"""
features from query itself, to document's bag-of-entities
    fill the 4 way matrix

a subclass of LeToRFeatureExtractor

features:
    tf weighted q-doc e textual similarities
    coverage of doc e's name on query
    e's rank in query's reference entity ranking (indri, FACC1)


"""


from knowledge4ir.feature import (
    LeToRFeatureExtractor,
    TermStat,
)
from traitlets import (
    Unicode,
    List,
    Int
)
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_corpus_stat,
    text2lm,
)
from knowledge4ir.utils import TARGET_TEXT_FIELDS


class LeToREntityRankingFeatureExtractorC(LeToRFeatureExtractor):
    feature_name_pre = Unicode('ERank')
    l_text_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    l_model = List(Unicode,
                   default_value=['lm_dir', 'bm25', 'coordinate', 'tf_idf']
                   ).tag(config=True)
    l_pooling = List(Unicode,
                     default_value=['tf', 'max']).tag(config=True)
    l_rank_feature = List(Unicode,
                          default_value=['err']
                          ).tag(config=True)
    l_e_field = List(Unicode,
                     default_value=['name', 'desp']).tag(config=True)
    corpus_stat_pre = Unicode(help="the file pre of corpus stats").tag(config=True)
    l_ref_rank = List(Unicode, help='query reference entity ranking').tag(config=True)
    l_ref_rank_name = List(Unicode, help='query reference rank name').tag(config=True)

    def __init__(self, **kwargs):
        super(LeToREntityRankingFeatureExtractorC, self).__init__(**kwargs)
        self.h_corpus_stat = {}
        self.h_field_df = {}
        self._load_corpus_stat()

    def _load_corpus_stat(self):
        l_field_h_df, self.h_corpus_stat = load_corpus_stat(
            self.corpus_stat_pre, self.l_text_fields)
        self.h_field_h_df = dict(l_field_h_df)
        for field in self.l_text_fields:
            assert field in self.h_corpus_stat
            assert field in self.h_field_h_df



    def extract(self, qid, docno, h_q_info, h_doc_info):
        return





