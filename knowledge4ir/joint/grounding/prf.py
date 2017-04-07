"""
prf grounding
vote the entity using prf's infor
starting with RM3's entities

features:
    max, mean, bin 1 and bin 2 of the RM3's entities' vote
"""


from knowledge4ir.joint.grounding import Grounder
import logging
import json
from traitlets import Int


class PrfGrounder(Grounder):
    nb_prf_e = Int(100, help='nb of prf entities to use').tag(config=True)

    def set_resource(self, external_resource):
        super(PrfGrounder, self).set_resource(external_resource)
        assert external_resource.h_q_boe_rm3

    def extract_for_entity(self, h_e_info, h_sf_info, h_info):
        h_root = super(PrfGrounder, self).extract_for_entity(h_e_info, h_sf_info, h_info)

        h_feature = dict()
        for key, score in h_root:
            if 'bin' in key:
                continue
            h_feature[key] = score

        e_id = h_e_info['id']
        # h_feature['e_cmns'] = h_e_info['cmns']
        l_rm3_e_score = self.resource.h_q_boe_rm3.get(h_info['qid'], [])
        l_rm3_e_score = l_rm3_e_score[:self.nb_prf_e]
        h_feature.update(self._prf_embedding_vote(e_id, l_rm3_e_score))
        logging.debug('e [%s] feature %s', e_id, json.dumps(h_feature))
        return h_feature

    def _prf_embedding_vote(self, e_id, l_rm3_e_score):

        l_sim = self._calc_sim_vec(e_id, [item[0] for item in l_rm3_e_score])
        l_weight = [item[1] for item in l_rm3_e_score]

        max_sim, mean_sim, l_bin = self._pool_sim_score(l_sim, l_weight)
        h_feature = dict()
        # h_feature['prf_vote_emb_max'] = max_sim
        h_feature['prf_vote_emb_mean'] = mean_sim
        # for i in xrange(len(l_bin)):
        #     h_feature['prf_vote_bin_%d' % i] = l_bin[i]
        return h_feature

    def _calc_sim_vec(self, e_id, l_e):
        l_sim = [int(e_id == e) for e in l_e]
        if e_id not in self.resource.embedding:
            return l_sim
        for i in xrange(len(l_e)):
            if l_e[i] == e_id:
                continue
            if l_e[i] not in self.resource.embedding:
                continue

            l_sim[i] = self.resource.embedding.similarity(e_id, l_e[i])

        return l_sim

