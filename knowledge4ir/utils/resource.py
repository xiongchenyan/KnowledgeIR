"""
resource to keep in memory to be shared across the pipeline
"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
    Int,
)
import logging
import json
from gensim.models import Word2Vec
from knowledge4ir.utils.retrieval_model import CorpusStat
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_json_info,
)
from knowledge4ir.utils.kg import (
    load_nlss_dict,
)


class JointSemanticResource(Configurable):
    surface_form_path = Unicode(help="the location of surface form dict, in Json format"
                                ).tag(config=True)
    embedding_path = Unicode(help="embedding location (word2vec format)"
                             ).tag(config=True)
    entity_embedding_path = Unicode(help="entity only embedding location word2vec format"
                                    ).tag(config=True)
    surface_stat_path = Unicode(help="the location of surface form stat dict in json"
                                ).tag(config=True)
    entity_field_path = Unicode(help="entity field path"
                                ).tag(config=True)
    boe_rm3_path = Unicode(help="boe rm3 trec rank format path"
                           ).tag(config=True)
    prf_sent_path = Unicode(help="prf sentence json dict path"
                            ).tag(config=True)
    entity_edge_path = Unicode(help='entity edge json path').tag(config=True)

    l_nlss_path = List(Unicode, help='paths to different nlss dumps').tag(config=True)
    l_nlss_name = List(Unicode, help='names of nlss').tag(config=True)
    max_nlss_per_e = Int(100, help='maximum nlss per e to derive').tag(config=True)

    entity_desp_path = Unicode(help='e id desp file').tag(config=True)
    def __init__(self, **kwargs):
        super(JointSemanticResource, self).__init__(**kwargs)
        self.embedding = None
        self.entity_embedding = None
        self.h_surface_form = None
        self.h_surface_stat = None
        self.h_entity_fields = None
        self.h_q_boe_rm3 = None
        self.h_q_prf_sent = None
        self.l_h_nlss = None
        self.h_e_edge = None
        self.h_e_desp = None
        self._load()
        self.corpus_stat = CorpusStat(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(JointSemanticResource, cls).class_print_help(inst)
        CorpusStat.class_print_help(inst)

    def _load(self):
        logging.info('start loading joint semantic resources')
        self._load_edge()
        self._load_nlss()
        self._load_entity_fields()
        self._load_sf()
        self._load_emb()
        self._load_e_emb()
        self._load_sf_stat()
        self._load_boe_rm3()
        self._load_prf_sent()
        self._load_desp()
        logging.info('joint semantic resource loaded')
        return

    def _load_desp(self):
        if not self.entity_desp_path:
            return
        logging.info('loading entity desp [%s]', self.entity_desp_path)
        h_nlss = load_nlss_dict(self.entity_desp_path, 1)
        for e, l_sent in h_nlss.items():
            self.h_e_desp[e] = l_sent[0][0]
        logging.info('loaded [%d] entity descriptions', len(self.h_e_desp))

    def _load_edge(self):
        if not self.entity_edge_path:
            return
        logging.info('loading entity edges from [%s]', self.entity_edge_path)
        self.h_e_edge = load_json_info(self.entity_edge_path, 'id')
        logging.info('[%d] entities\'s edge loaded', len(self.h_e_edge))

    def _load_nlss(self):
        if not self.l_nlss_path:
            return
        assert len(self.l_nlss_path) == len(self.l_nlss_name)
        logging.info('nlss: %s',
                     json.dumps(zip(self.l_nlss_name, self.l_nlss_path)))
        self.l_h_nlss = [load_nlss_dict(nlss_path, self.max_nlss_per_e)
                         for nlss_path in self.l_nlss_path]
        logging.info('nlss loaded')

    def _load_sf(self):
        if not self.surface_form_path:
            return
        logging.info('loading sf dict from [%s]', self.surface_form_path)
        self.h_surface_form = json.load(open(self.surface_form_path))
        logging.info('sf dict of [%d] size loaded', len(self.h_surface_form))

    def _load_sf_stat(self):
        if not self.surface_stat_path:
            return
        logging.info('loading sf stat from [%s]', self.surface_stat_path)
        self.h_surface_stat = json.load(open(self.surface_stat_path))
        logging.info('sf stat of [%d] size loaded', len(self.h_surface_stat))

    def _load_emb(self):
        if not self.embedding_path:
            return
        logging.info('loading embedding [%s]', self.embedding_path)
        self.embedding = Word2Vec.load_word2vec_format(self.embedding_path)
        logging.info('embedding loaded')

    def _load_e_emb(self):
        if not self.entity_embedding_path:
            return
        logging.info('loading entity embedding [%s]', self.entity_embedding_path)
        self.entity_embedding = Word2Vec.load_word2vec_format(self.entity_embedding_path)
        logging.info('entity embedding loaded')

    def _load_entity_fields(self):
        if not self.entity_field_path:
            return
        logging.info('loading entity fields from [%s]', self.entity_field_path)
        self.h_entity_fields = dict()
        for line in open(self.entity_field_path):
            h_field = json.loads(line)
            self.h_entity_fields[h_field['id']] = h_field

        logging.info('total [%d] entity fields loaded', len(self.h_entity_fields))

    def _load_boe_rm3(self):
        if not self.boe_rm3_path:
            return
        l_q_e_score = load_trec_ranking_with_score(self.boe_rm3_path)
        self.h_q_boe_rm3 = dict(l_q_e_score)
        return

    def _load_prf_sent(self):
        if not self.prf_sent_path:
            return
        self.h_q_prf_sent = json.load(open(self.prf_sent_path))
        return