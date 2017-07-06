"""
nlss grid feature
"""
import json

import numpy as np
from traitlets import Unicode, List

from knowledge4ir.boe_exact.nlss_feature import NLSSFeature
from knowledge4ir.utils import E_GRID_FIELD, add_feature_prefix, SPOT_FIELD


class EGridNLSSFeature(NLSSFeature):
    """
    extract boe exact features by comparing e_grid of qe with qe's nlss
    """
    feature_name_pre = Unicode('EGridNLSS')
    l_grid_lvl_pool = List(Unicode, default_value=['Sum', 'Max', 'Mean']).tag(config=True)
    h_pool_name_func = {
        'Sum': np.sum,
        'Max': np.amax,
        'Mean': np.mean,
    }

    def __init__(self, **kwargs):
        super(EGridNLSSFeature, self).__init__(**kwargs)
        for pool_name in self.l_grid_lvl_pool:
            assert pool_name in self.h_pool_name_func

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
        """
        :param ana:
        :param doc_info:
        :param l_qe_nlss:
        :return:
        """

        h_this_feature = dict()
        h_e_grid = doc_info.get(E_GRID_FIELD, {})
        l_nlss_bow = self._form_nlss_bow(l_qe_nlss)
        l_nlss_emb = self._form_nlss_emb(l_qe_nlss)
        for field in self.l_target_fields:
            if field not in h_e_grid:
                continue
            l_e_grid = h_e_grid.get(field, [])
            h_field_grid_feature = self._extract_per_entity_per_nlss_per_field(
                ana, doc_info, l_qe_nlss, l_e_grid, l_nlss_bow, l_nlss_emb)
            h_this_feature.update(add_feature_prefix(h_field_grid_feature, field + '_'))
        return h_this_feature

    def _extract_per_entity_per_nlss_per_field(
            self, ana, doc_info, l_qe_nlss, l_e_grid,  l_nlss_bow, l_nlss_emb):
        """
        for each sentence in e_grid,
            check if ana e in it, and if len < max_sent_len
            calculate similarity with all qe_nlss
            average and max sum up
        :param ana:
        :param doc_info:
        :param l_qe_nlss: nlss of qe
        :param l_e_grid: grid of this field
        :param l_nlss_bow: pre calc bow of nlss
        :param l_nlss_emb: pre calc emb of nlss
        :return:
        """
        e_id = ana['id']
        l_this_e_grid = self._filter_e_grid(e_id, l_e_grid)
        l_grid_bow = self._form_grid_bow(l_this_e_grid)
        l_grid_emb = self._form_grid_emb(l_this_e_grid)

        m_bow_sim = self._calc_bow_trans(l_grid_bow, l_nlss_bow)
        m_emb_sim = self._calc_emb_trans(l_grid_emb, l_nlss_emb)

        self._log_intermediate_res(ana, doc_info, l_this_e_grid, l_qe_nlss, m_bow_sim, m_emb_sim)

        h_bow_feature = self._pool_grid_nlss_sim(m_bow_sim)
        h_emb_feature = self._pool_grid_nlss_sim(m_emb_sim)

        h_feature = dict()
        h_feature.update(add_feature_prefix(h_bow_feature, 'BOW_'))
        h_feature.update(add_feature_prefix(h_emb_feature, 'Emb_'))
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
            if len(e_grid['sent'].split()) > self.max_sent_len:
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

    def _pool_grid_nlss_sim(self, trans_mtx):
        h_feature = {}
        for name1 in self.l_grid_lvl_pool:
            f1 = self.h_pool_name_func[name1]
            for f2, name2 in zip([np.mean, np.amax], ['Mean', 'Max']):
                score = -1
                if (trans_mtx.shape[0] > 0) & (trans_mtx.shape[1] > 0):
                    score = f1(f2(trans_mtx, axis=1), axis=0)
                pool_name = 'R' + name1 + 'C' + name2
                h_feature[pool_name] = score
        return h_feature

    def _log_intermediate_res(self, ana, doc_info, l_this_e_grid, l_qe_nlss, m_bow_sim, m_emb_sim):
        """
        dump out the intermediate results
            e_id, name, doc no,
            e_grid_sentences:
            grid sentence for this e_id in doc, mean sim in bow and emb,
                max sim in bow and emb, and corresponding nlss that generate the max
        :param ana:
        :param doc_info:
        :param l_this_e_grid:
        :param l_qe_nlss:
        :param m_bow_sim:
        :param m_emb_sim:
        :return:
        """
        # use json
        if not doc_info:
            return
        h_pair_res = dict()
        h_pair_res['id'] = ana['id']
        h_pair_res['surface'] = ana['surface']
        h_pair_res['docno'] = doc_info['docno']
        if (not l_this_e_grid) | (not l_qe_nlss):
            print >> self.intermediate_out, json.dumps(h_pair_res)
            return

        l_e_grid_info = []
        for i in xrange(len(l_this_e_grid)):
            h_this_sent = {}
            h_this_sent['sent'] = l_this_e_grid[i]['sent']
            h_this_sent['mean_bow_sim'] = np.mean(m_bow_sim[i])
            h_this_sent['mean_emb_sim'] = np.mean(m_emb_sim[i])

            max_p = np.argmax(m_bow_sim[i])
            h_this_sent['max_bow_sim'] = m_bow_sim[i, max_p]
            h_this_sent['max_bow_nlss'] = l_qe_nlss[max_p][0]
            max_p = np.argmax(m_emb_sim[i])
            h_this_sent['max_emb_sim'] = m_emb_sim[i, max_p]
            h_this_sent['max_emb_nlss'] = l_qe_nlss[max_p][0]

            l_e_grid_info.append(h_this_sent)

        h_pair_res['e_grid_info'] = l_e_grid_info

        print >> self.intermediate_out, json.dumps(h_pair_res)
        return