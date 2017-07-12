"""
Entity Anchored Semi-structure doc representation
    to get better LeToR features

In the pipeline, form as a subclass of NLSSFeature

do:
    construct entity Anchored Semi-structure representation
    sentence -> entity, only if query entity is in the sentence
        the scores include:
            embedding sim
            bow sim (frequency)
            embedding sim vs desp
            bow sim vs desp

    extract features from the constructed grid:
        pooling grid scores to features
        entity proximity
        word proximity (another class)
        full letor (another class)
"""

from traitlets import (
    Unicode,
    Int,
    List,
)
import numpy as np
from scipy.spatial.distance import cosine
from knowledge4ir.boe_exact.boe_feature import BoeFeature
from knowledge4ir.utils import (
    QUERY_FIELD,
    TARGET_TEXT_FIELDS,
    lm_cosine,
    avg_embedding,
    body_field,
    E_GRID_FIELD,
    add_feature_prefix,
    SPOT_FIELD,
    term2lm,
    text2lm,
    mean_pool_feature,
    max_pool_feature,
)
from knowledge4ir.utils.retrieval_model import RetrievalModel
import logging
import json


class EntityAnchorFeature(BoeFeature):
    feature_name_pre = Unicode('EA')
    l_target_fields = List(Unicode, default_value=[body_field]).tag(config=True)
    gloss_len = Int(15, help='gloss length').tag(config=True)
    max_grid_sent_len = Int(100, help='max grid sentence len to consider').tag(config=True)
    l_grid_scores = ['freq', 'uw_emb', 'gloss_emb', 'gloss_bow', 'desp_emb', 'desp_bow']
    l_feature = List(Unicode, default_value=['passage', 'grid']).tag(config=True)

    def set_resource(self, resource):
        self.resource = resource
        assert self.resource.h_e_desp
        assert self.resource.embedding

    def extract_per_entity(self, q_info, ana, doc_info):
        """
        extract per entity feature
        :param q_info:
        :param ana:
        :param doc_info:
        :return:
        """
        h_feature = {}
        qe = ana['id']
        for field in self.l_target_fields:
            l_grid = doc_info.get(E_GRID_FIELD, {}).get(field, [])
            l_grid = self._filter_e_grid(qe, l_grid)
            l_grid = self._calc_grid_scores(l_grid)
            if 'passage' in self.l_feature:
                h_proximity_f = self._entity_proximity_features(q_info, l_grid, field)
                h_feature.update(add_feature_prefix(h_proximity_f, field + '_'))
            if 'grid' in self.l_feature:
                h_grid_score_f = self._grid_score_features(qe, l_grid)
                h_feature.update(add_feature_prefix(h_grid_score_f, field + '_'))
        return h_feature

    def _calc_grid_scores(self, l_grid):
        """
        sent -> e scores
        include:
            frequency:
            emb_sim:
            desp_emb:
            desp_bow:
            gloss_emb:
            gloss_bow:
        :param l_grid:
        :return: for grid->'entity'->['id': e id, 'name':score], grid_score = {name:score}
        """
        logging.info('start calculating grid scores')
        for grid in l_grid:
            l_e = [ana['id'] for ana in grid.get(SPOT_FIELD)]
            h_e_tf = term2lm(l_e)
            grid_sent = grid['sent']
            grid_lm = text2lm(grid_sent)
            grid_emb = avg_embedding(self.resource.embedding, grid_sent)

            l_e_score = []
            for e, tf in h_e_tf.items():
                h_e_score = {'id': e, 'freq': tf}
                h_e_score['uw_emb'] = self._e_grid_emb(e, grid_emb)
                h_e_score['gloss_emb'] = self._e_gloss_emb(e, grid_emb)
                h_e_score['gloss_bow'] = self._e_gloss_bow(e, grid_lm)
                h_e_score['desp_emb'] = self._e_desp_emb(e, grid_emb)
                h_e_score['desp_bow'] = self._e_desp_bow(e, grid_lm)
                l_e_score.append(h_e_score)
            grid['e_score'] = l_e_score

        return l_grid

    def _e_grid_emb(self, e, grid_emb):
        if e not in self.resource.embedding:
            return 0
        e_emb = self.resource.embedding[e]
        if grid_emb is None:
            return 0
        return 1 - cosine(e_emb, grid_emb)

    def _e_gloss_emb(self, e, grid_emb):
        desp = self.resource.h_e_desp.get(e, "")
        gloss = ' '.join(desp.split()[:self.gloss_len])
        e_emb = avg_embedding(self.resource.embedding, gloss)
        if (e_emb is None) | (grid_emb is None):
            return 0
        return max(1 - cosine(e_emb, grid_emb), 0)

    def _e_desp_emb(self, e, grid_emb):
        desp = self.resource.h_e_desp.get(e, "")
        e_emb = avg_embedding(self.resource.embedding, desp)
        if (e_emb is None) | (grid_emb is None):
            return 0
        return max(1 - cosine(e_emb, grid_emb), 0)

    def _e_gloss_bow(self, e, grid_lm):
        desp = self.resource.h_e_desp.get(e, "")
        gloss = ' '.join(desp.split()[:self.gloss_len])
        e_lm = text2lm(gloss)
        return lm_cosine(e_lm, grid_lm)

    def _e_desp_bow(self, e, grid_lm):
        desp = self.resource.h_e_desp.get(e, "")
        e_lm = text2lm(desp)
        return lm_cosine(e_lm, grid_lm)

    def _entity_proximity_features(self, q_info, l_grid, field):
        l_grid_sent = [grid['sent'] for grid in l_grid]
        l_grid_lm = [text2lm(sent) for sent in l_grid_sent]
        q_lm = text2lm(q_info['query'])
        l_scores = []
        for grid_lm in l_grid_lm:
            r_model = RetrievalModel()
            r_model.set_from_raw(
                q_lm, grid_lm,
                self.resource.corpus_stat.h_field_df.get(field, None),
                self.resource.corpus_stat.h_field_total_df.get(field, None),
                self.resource.corpus_stat.h_field_avg_len.get(field, None)
            )
            l_scores.append(dict(r_model.scores()))
        h_feature = dict()
        # h_feature.update(mean_pool_feature(l_scores))
        h_feature.update(max_pool_feature(l_scores))

        grid_lm = text2lm(' '.join(l_grid_sent))
        r_model = RetrievalModel()
        r_model.set_from_raw(
            q_lm, grid_lm,
            self.resource.corpus_stat.h_field_df.get(field, None),
            self.resource.corpus_stat.h_field_total_df.get(field, None),
            self.resource.corpus_stat.h_field_avg_len.get(field, None)
        )
        h_score = dict(r_model.scores())
        h_feature.update(h_score)

        h_feature = add_feature_prefix(h_feature, 'EntityProximity')
        return h_feature

    def _grid_score_features(self, qe, l_grid):
        """

        :param qe:
        :param l_grid:
        :return: h_feature
        """
        h_feature = dict()
        ll_grid_e_score = [[e_score for e_score in grid['e_score'] if e_score['id'] == qe]
                           + [e_score for e_score in grid['e_score'] if e_score['id'] != qe]
                           for grid in l_grid]
        for name in self.l_grid_scores:
            ll_this_grid_score = [[h_score.get(name, 0) for h_score in l_grid_e_score]
                                  for l_grid_e_score in ll_grid_e_score
                                  ]
            h_this_score = dict()
            h_this_score['Sum'] = sum([l_score[0] for l_score in ll_this_grid_score])
            h_this_score['Max'] = max([l_score[0] for l_score in ll_this_grid_score])
            h_this_score['FullCombine'] = sum(
                sum(ll_this_grid_score, []),
                [sum(l_score) / max(float(len(l_score)), 1.0) for l_score in ll_this_grid_score]
            )
            h_this_score['NormSum'] = sum([l_score[0] / float(max(sum(l_score), 1))
                                           for l_score in ll_this_grid_score])
            h_this_score = add_feature_prefix(h_this_score, name)
            h_feature.update(h_this_score)
        return h_feature

    def _filter_e_grid(self, e_id, l_e_grid):
        """
        filer e grid to those that
            contain e id
            not too long (<self.max_sent_len)
        :param e_id: target e id
        :param l_e_grid: grid of doc
        :return:
        """
        l_kept_grid = []
        for e_grid in l_e_grid:
            if len(e_grid['sent'].split()) > self.max_grid_sent_len:
                continue
            contain_flag = False
            for ana in e_grid[SPOT_FIELD]:
                if ana['id'] == e_id:
                    contain_flag = True
                    break
            if contain_flag:
                l_kept_grid.append(e_grid)
        return l_kept_grid

    def _form_grid_bow(self, l_e_grid):
        l_sent = [grid['sent'] for grid in l_e_grid]
        return self._form_sents_bow(l_sent)

    def _form_grid_emb(self, l_e_grid):
        l_sent = [grid['sent'] for grid in l_e_grid]
        return self._form_sents_emb(l_sent)