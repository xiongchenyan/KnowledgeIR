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
    l_target_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    gloss_len = Int(15, help='gloss length').tag(config=True)
    max_grid_sent_len = Int(100, help='max grid sentence len to consider').tag(config=True)
    l_grid_scores = ['freq', 'uw_emb', 'desp_emb', 'desp_bow']
    l_feature = List(Unicode, default_value=['passage', 'grid', 'coherence', 'desp']).tag(config=True)

    def set_resource(self, resource):
        self.resource = resource
        # assert self.resource.h_e_desp
        # assert self.resource.embedding

    def extract_pair(self, q_info, doc_info):
        h_feature = super(EntityAnchorFeature, self).extract_pair(q_info, doc_info)
        if 'coherence' in self.l_feature:
            h_global_coherence = self._global_grid_coherence(doc_info)
            h_feature.update(add_feature_prefix(h_global_coherence, self.feature_name_pre))
        return h_feature

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
            l_qe_grid = self._filter_e_grid(qe, l_grid)
            if 'grid' in self.l_feature:
                l_qe_grid = self._calc_grid_scores(l_qe_grid)

            if 'passage' in self.l_feature:
                h_proximity_f = self._entity_passage_features(q_info, l_qe_grid, field)
                h_feature.update(add_feature_prefix(h_proximity_f, field + '_'))
            if 'desp' in self.l_feature:
                h_desp_f = self._desp_passage_features(qe, l_qe_grid, field)
                h_feature.update(add_feature_prefix(h_desp_f, field + '_'))
            if 'grid' in self.l_feature:
                h_grid_score_f = self._grid_score_features(qe, l_qe_grid)
                h_feature.update(add_feature_prefix(h_grid_score_f, field + '_'))
            if 'coherence' in self.l_feature:
                if field == body_field:
                    h_coherence_f = self._qe_grid_coherence(qe, l_grid)
                    h_feature.update(add_feature_prefix(h_coherence_f, field + '_'))
        return h_feature

    def _qe_grid_coherence(self, qe, l_grid):
        h_feature = {}
        h_feature.update(self._single_e_coherence(qe, l_grid))
        h_feature.update(self._pair_e_coherence(l_grid, qe))
        h_feature = add_feature_prefix(h_feature, 'Qe')
        return h_feature

    def _global_grid_coherence(self, doc_info):
        if body_field in self.l_target_fields:
            l_grid = doc_info.get(E_GRID_FIELD, {}).get(body_field, [])
            return self._pair_e_coherence(l_grid)

    def _single_e_coherence(self, e_id, l_grid):
        h_feature = dict()

        h_e_pos = self._form_grid_reverse_index(l_grid)
        if len(h_e_pos.get(e_id, [])) < 2:
            h_feature['uniLC'] = 0
            return h_feature

        l_uni_lc = []
        l_pos = [0] + h_e_pos[e_id] + [len(l_grid)]

        for i in xrange(1, len(l_pos) - 1):
            lc = min(l_pos[i] - l_pos[i - 1], l_pos[i + 1] - l_pos[i - 1])
            lc = 1.0 / max(lc, 1.0)
            l_uni_lc.append(lc)
        uniLC = sum(l_uni_lc) / float(len(l_uni_lc))
        h_feature['uniLC'] = uniLC
        return h_feature

    def _pair_e_coherence(self, l_grid, target_e=None):
        h_feature = dict()
        h_e_pos = self._form_grid_reverse_index(l_grid)
        l_p = range(len(l_grid))
        if target_e is not None:
            l_p = h_e_pos.get(target_e)
        l_total_bipLC = []
        for grid_p in l_p:
            grid = l_grid[grid_p]
            l_e = list(set([ana['id'] for ana in grid['spot']]))
            bipLC = 0
            cnt = 0
            for i in xrange(len(l_e)):
                set_i = set(h_e_pos.get(l_e[i], []))
                for j in xrange(i + 1, len(l_e)):
                    if target_e is not None:
                        if (l_e[i] != target_e) & (l_e[j] != target_e):
                            continue
                    cnt += 1
                    set_j = set(h_e_pos.get(l_e[j], []))
                    l_both_p = list(set_i.intersection(set_j))
                    if len(l_both_p) < 2:
                        continue
                    dist = min([abs(p - grid_p) for p in l_both_p if p != grid_p])
                    bipLC += 1.0 / dist
            bipLC /= max(cnt, 1.0)
            l_total_bipLC.append(bipLC)
        avg_bipLC = 0
        if len(l_total_bipLC):
            avg_bipLC = sum(l_total_bipLC) / float(len(l_total_bipLC))
        h_feature['bipLC'] = avg_bipLC
        return h_feature

    def _form_grid_reverse_index(self, l_grid):
        h_e_pos = dict()
        for pos, grid in enumerate(l_grid):
            for ana in grid.get('spot'):
                e_id = ana['id']
                if e_id not in h_e_pos:
                    h_e_pos[e_id] = []
                h_e_pos[e_id].append(pos)
        return h_e_pos

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
                # h_e_score['gloss_emb'] = self._e_gloss_emb(e, grid_emb)
                # h_e_score['gloss_bow'] = self._e_gloss_bow(e, grid_lm)
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

    def _entity_passage_features(self, q_info, l_grid, field):
        l_grid_sent = [grid['sent'] for grid in l_grid]
        q_lm = text2lm(q_info['query'])
        h_feature = dict()
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

        # l_grid_lm = [text2lm(sent) for sent in l_grid_sent]
        # l_scores = []
        # for grid_lm in l_grid_lm:
        #     r_model = RetrievalModel()
        #     r_model.set_from_raw(
        #         q_lm, grid_lm,
        #         self.resource.corpus_stat.h_field_df.get(field, None),
        #         self.resource.corpus_stat.h_field_total_df.get(field, None),
        #         self.resource.corpus_stat.h_field_avg_len.get(field, None)
        #     )
        #     l_scores.append(dict(r_model.scores()))
        # # h_feature.update(mean_pool_feature(l_scores))
        # h_feature.update(max_pool_feature(l_scores))

        h_feature = add_feature_prefix(h_feature, 'EntityPassage')
        return h_feature

    def _desp_passage_features(self, e_id, l_grid, field):
        l_grid_sent = [grid['sent'] for grid in l_grid]
        q_lm = text2lm(self.resource.h_e_desp.get(e_id, ""))
        grid_lm = text2lm(' '.join(l_grid_sent))
        r_model = RetrievalModel()
        r_model.set_from_raw(
            q_lm, grid_lm,
            self.resource.corpus_stat.h_field_df.get(field, None),
            self.resource.corpus_stat.h_field_total_df.get(field, None),
            self.resource.corpus_stat.h_field_avg_len.get(field, None)
        )
        h_score = dict(r_model.scores())
        h_feature = add_feature_prefix(h_score, 'DespPassage')
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
            l_qe_score = [l_score[0] for l_score in ll_this_grid_score]
            if not l_qe_score:
                l_qe_score.append(0)
            h_this_score['Sum'] = sum(l_qe_score)
            h_this_score['Max'] = max(l_qe_score)
            # h_this_score['FullCombine'] = sum(
            #     [sum(l_score) / max(float(len(l_score)), 1.0) for l_score in ll_this_grid_score]
            # )
            h_this_score['NormSum'] = sum([l_score[0] / float(max(sum(l_score), 1.0))
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