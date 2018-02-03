import torch
import torch.nn as nn


class Debugger:
    @staticmethod
    def show_evidence(mtx_e, mtx_embedding,
                      h_ent, h_evm, entity_range,
                      kp_mtx, knrm_res):
        target_emb = nn.functional.normalize(mtx_embedding, p=2,
                                             dim=-1)

        trans_mtx = torch.matmul(target_emb,
                                 target_emb.transpose(-2, -1))

        import sys
        for i in range(mtx_e.size()[0]):
            print 'Next document.'
            row = mtx_e[i]
            nodes = row.cpu().data.numpy().tolist()
            entities = [h_ent[n] for n in nodes if n < entity_range]
            events = [h_evm[n - entity_range] for n in nodes if
                      n >= entity_range]
            num_ent = len(entities)
            voting_scores = trans_mtx[i]

            event_voting_scores = voting_scores[
                                  num_ent:].cpu().data.numpy()
            event_kernel_scores = kp_mtx[i][
                                  num_ent:].cpu().data.numpy().tolist()
            event_final_scores = knrm_res[i][
                                 num_ent:].cpu().data.numpy()

            k = min(10, len(event_final_scores))
            topk = event_final_scores.argpartition(-k)[-k:].tolist()
            topk.reverse()

            # print [events[k] for k in topk]
            # print [event_kernel_scores[k] for k in topk]
            # print [event_final_scores.tolist()[k] for k in topk]
            print "Top K events"
            print event_final_scores[topk]
            for k in topk:
                print 'Showing: event %d: %s' % (k, events[k])
                print 'final score %.5f' % event_final_scores[k]
                print 'kernel scores: '
                print event_kernel_scores[k]
                votes = event_voting_scores[k]

                k = min(20, len(votes))
                top_votes = votes.argpartition(-k)[-k:].tolist()
                k = min(10, len(votes))
                bottom_votes = votes.argpartition(k)[:k].tolist()

                top_votes_node = [
                    h_evm[nodes[n] - entity_range] if n >= num_ent else
                    h_ent[nodes[n]] for n in top_votes
                ]

                bottom_votes_node = [
                    h_evm[nodes[n] - entity_range] if n >= num_ent else
                    h_ent[nodes[n]] for n in bottom_votes
                ]

                from operator import itemgetter
                print 'top votes'
                print sorted(zip(top_votes_node, votes[top_votes]),
                             key=itemgetter(1), reverse=True)

                print 'bottom votes'
                print sorted(zip(bottom_votes_node, votes[bottom_votes])
                             , key=itemgetter(1), reverse=True)

                sys.stdin.readline()
