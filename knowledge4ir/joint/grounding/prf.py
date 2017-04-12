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
from knowledge4ir.utils.nlp import avg_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


class PrfGrounder(Grounder):
    nb_prf_e = Int(0, help='nb of prf entities to use').tag(config=True)
    nb_prf_sent = Int(0, help='number of prf sentences to use').tag(config=True)

    def set_resource(self, external_resource):
        super(PrfGrounder, self).set_resource(external_resource)
        assert external_resource.embedding
        if self.nb_prf_e:
            assert external_resource.h_q_boe_rm3
        if self.nb_prf_sent:
            assert external_resource.h_q_prf_sent

    def extract_for_entity(self, h_e_info, h_sf_info, h_info):
        h_feature = super(PrfGrounder, self).extract_for_entity(h_e_info, h_sf_info, h_info)

        e_id = h_e_info['id']
        qid = h_info['qid']
        if self.nb_prf_e:
            l_rm3_e_score = self.resource.h_q_boe_rm3.get(qid, [])
            l_rm3_e_score = l_rm3_e_score[:self.nb_prf_e]
            h_feature.update(self._prf_embedding_vote(e_id, l_rm3_e_score))
        if self.nb_prf_sent:
            l_prf_sent = self.resource.h_q_prf_sent.get(qid, [])
            l_prf_sent = l_prf_sent[:self.nb_prf_sent]
            h_feature.update(self._prf_sent_vote(e_id, l_prf_sent))
        if self.nb_prf_sent + self.nb_prf_e:
            logging.debug('e [%s] added prf feature %s', e_id, json.dumps(h_feature))
        return h_feature

    def _prf_sent_vote(self, e_id, l_prf_sent_score):
        max_sim, mean_sim, w_mean_sim = 0, 0, 0
        if e_id in self.resource.embedding:
            e_emb = self.resource.embedding[e_id]
            l_score = [item[-1] for item in l_prf_sent_score]
            l_sent = [item[1] for item in l_prf_sent_score]
            l_sent_emb = [avg_embedding(self.resource.embedding, sent) for sent in l_sent]

            l_sim = []
            for sent_emb in l_sent_emb:
                if sent_emb is None:
                    l_sim.append(0)
                else:
                    sim = cosine_similarity(e_emb.reshape(1, -1), sent_emb.reshape(1, -1)).reshape(-1)[0]
                    l_sim.append(sim)

            v_w = np.array(l_score)  # make sure these are all positive
            v_w = v_w / np.sum(v_w)
            v_sim = np.array(l_sim)
            max_sim = np.max(v_sim)
            mean_sim = np.mean(v_sim)
            w_mean_sim = v_sim.dot(v_w)

        h_feature = {"prf_sent_vote_max": max_sim,
                     "prf_sent_vote_mean": mean_sim,
                     "prf_sent_vote_wmean": w_mean_sim,
                     }
        return h_feature

    def _prf_embedding_vote(self, e_id, l_rm3_e_score):

        l_sim = self._calc_sim_vec(e_id, [item[0] for item in l_rm3_e_score])
        l_weight = [item[1] for item in l_rm3_e_score]

        max_sim, mean_sim, l_bin = self._pool_sim_score(l_sim, l_weight)
        h_feature = dict()
        h_feature['prf_vote_emb_max'] = max_sim
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

