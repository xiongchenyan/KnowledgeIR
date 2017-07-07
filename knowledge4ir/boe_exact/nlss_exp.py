"""
q expansion with nlss
"""
import json

from traitlets import Int, Unicode

from knowledge4ir.boe_exact.nlss_feature import NLSSFeature
from knowledge4ir.utils import text2lm, max_pool_feature, add_feature_prefix, QUERY_FIELD, avg_embedding
from knowledge4ir.utils.retrieval_model import RetrievalModel


class NLSSExpansionFeature(NLSSFeature):
    """
    find best nlss for the query (using embedding cosine)
        top 5 for now
    and then use them to rank the document via qe-dw
        top 5 nlss combined as e-desp-alike big query
        each nlss individually, and then take a max
    also dump the top k nlss used
    """
    top_k_nlss = Int(5, help='number of nlss to use per query entity').tag(config=True)
    feature_name_pre = Unicode('NLSSExp')

    def _extract_per_entity_via_nlss(self, q_info, ana, doc_info, l_qe_nlss):
        """
        extract e-d features

        do:
            get top k nlss
            form doc lm
            retrieval, as a whole of individually
            sum up to features
        :param q_info: query info
        :param ana:
        :param doc_info:
        :param l_qe_nlss:
        :return: h_feature: entity features for this nlss set
        """

        l_top_nlss = self._find_top_k_nlss_for_q(q_info, ana, l_qe_nlss)

        l_top_sent = [nlss[0] for nlss in l_top_nlss]
        l_top_sent.append(' '.join(l_top_sent))
        if not l_top_sent:
            l_top_sent.append('')  # place holder for empty nlss e
        l_h_per_sent_feature = []
        l_field_doc_lm = [text2lm(doc_info.get(field, ""), clean=True)
                          for field in self.l_target_fields]
        for sent in l_top_sent:
            h_per_sent_feature = {}
            h_sent_lm = text2lm(sent, clean=True)
            for field, lm in zip(self.l_target_fields, l_field_doc_lm):
                r_model = RetrievalModel()
                r_model.set_from_raw(
                    h_sent_lm, lm,
                    self.resource.corpus_stat.h_field_df.get(field, None),
                    self.resource.corpus_stat.h_field_total_df.get(field, None),
                    self.resource.corpus_stat.h_field_avg_len.get(field, None)
                )
                l_retrieval_score = r_model.scores()
                q_len = float(max(sum([item[1] for item in h_sent_lm.items()]), 1))

                h_per_sent_feature.update(dict(
                    [(field + name, score / q_len) for name, score in l_retrieval_score]
                ))
            l_h_per_sent_feature.append(h_per_sent_feature)

        h_max_feature = max_pool_feature(l_h_per_sent_feature[:-1])
        h_mean_feature = add_feature_prefix(l_h_per_sent_feature[-1], 'Conca')

        h_feature = h_max_feature
        h_feature.update(h_mean_feature)
        return h_feature

    def _find_top_k_nlss_for_q(self, q_info, ana, l_qe_nlss):
        """
        find top k similar sentences based on cosine(q emb, sent emb)
        :param q_info: query info
        :param ana: current q e
        :param l_qe_nlss: nlss of this e
        :return:
        """

        query = q_info[QUERY_FIELD]
        q_emb = avg_embedding(self.resource.embedding, query)
        l_nlss_emb = self._form_nlss_emb(l_qe_nlss)

        m_emb_sim = self._calc_emb_trans([q_emb], l_nlss_emb)
        l_emb_sim_score = m_emb_sim[0].tolist()
        l_nlss_with_score = zip(l_qe_nlss, l_emb_sim_score)
        l_nlss_with_score.sort(key=lambda item: item[1], reverse=True)
        l_top_nlss = [item[0] for item in l_nlss_with_score[:self.top_k_nlss]]

        self._log_qe_top_nlss(q_info, ana, l_top_nlss)

        return l_top_nlss

    def _log_qe_top_nlss(self, q_info, ana, l_top_nlss):
        """
        dump a packed intermediate information in it
        """
        h_info = dict(q_info)
        h_info['current_e'] = ana
        h_info['top_nlss'] = l_top_nlss
        print >> self.intermediate_out, json.dumps(h_info)