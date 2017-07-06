"""
nlss edge count features

calculate the # of doc entities that share the same nlss
and also
    # of doc entities share a real freebase edge
    # of doc entities that has a virtual embedding edge
        cosine(emb, emb) > 0.2
"""

import json
import logging
from traitlets import Unicode

from knowledge4ir.boe_exact.nlss_feature import NLSSFeature
from knowledge4ir.utils import E_GRID_FIELD, add_feature_prefix, SPOT_FIELD
from knowledge4ir.utils.boe import form_boe_per_field


class NLSSEdgeCountFeature(NLSSFeature):
    feature_name_pre = Unicode('EdgeCnt')

    def set_resource(self, resource):
        super(NLSSEdgeCountFeature, self).set_resource(resource)
        assert self.resource.h_e_edge

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
        """
        for each field
            # of e share nlss
            # of e connected in Freebase
            # of e emb cosine > 0.2
        :param q_info:
        :param ana: the current query ana e
        :param doc_info:
        :param l_qe_nlss:
        :return:
        """
        qe = ana['id']
        h_feature = {}
        for field in self.l_target_fields:
            l_e = form_boe_per_field(doc_info, field)
            l_e = [e['id'] for e in l_e if e['id'] != qe]

            nlss_cnt = self._count_co_nlss(qe, l_e, l_qe_nlss)
            emb_sim_cnt = self._count_meaningful_emb_sim(qe, l_e)
            kg_edge_cnt = self._count_kg_edge(qe, l_e)

            h_feature['nlss_cnt'] = nlss_cnt
            h_feature['emb_sim_cnt'] = emb_sim_cnt
            h_feature['kg_edge_cnt'] = kg_edge_cnt
            h_feature['nb_e'] = len(l_e)
            add_feature_prefix(h_feature, field)
        return h_feature

    def _count_co_nlss(self, qe, l_e, l_qe_nlss):
        """
        just check # of e in l_e that appear in l_qe_nlss[(sent, l_e)]
        :param qe:
        :param l_e:
        :param l_qe_nlss:
        :return:
        """

        l_nlss_e = sum(
            [ nlss[1] for nlss in l_qe_nlss],
            [])
        s_nlss_e = set(l_nlss_e)
        logging.info('[%s] has [%d] total e connected in nlss', qe, len(s_nlss_e))

        nlss_cnt = len([e for e in l_e if e in s_nlss_e])
        logging.info('[%d] doc e, [%d] connected in nlss to [%s]',
                     len(l_e), nlss_cnt, qe)
        return nlss_cnt

    def _count_meaningful_emb_sim(self, qe, l_e):
        """

        :param qe:
        :param l_e:
        :return:
        """

        if qe not in self.resource.embedding:
            return

        emb_cnt = 0
        for e in l_e:
            if e not in self.resource.embedding:
                continue
            sim = self.resource.embedding.similarity(qe, e)
            if sim >= 0.2:
                emb_cnt += 1
        return emb_cnt

    def _count_kg_edge(self, qe, l_e):
        l_edge = self.resource.h_e_edge.get(qe, {}).get('edges', [])
        s_qe_neighbor = [item[1] for item in l_edge if item[1].startswith("/m/")]
        logging.info('[%s] has [%d] neighbor', qe, len(s_qe_neighbor))
        kg_cnt = 0
        for e in l_e:
            if e in s_qe_neighbor:
                kg_cnt += 1
        logging.info('[%d] in doc', kg_cnt)
        return kg_cnt



