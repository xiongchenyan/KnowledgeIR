"""
pretty comparison of e attentions from two methods


input:
    eval of a
    eval of b
    e att of a and b (the output of append_tesxt_to_e_att's results)

output:
    qid \t query \t ndcg a \t ndcg b sf \t e_1 \t name \t score from a \t score from b
    \t \t \t \t \t other entity
    \t\t\t other spots
"""

from knowledge4ir.utils import load_gdeval_res
import json
import logging
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List,
)
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


class PrettyCompEAtt(Configurable):
    l_eval_in = List(Unicode, help='2 evaluation res').tag(config=True)
    l_e_att_in = List(Unicode, help='2 att res').tag(config=True)
    out_name = Unicode(help='out name').tag(config=True)

    def __init__(self, **kwargs):
        super(PrettyCompEAtt, self).__init__(**kwargs)
        self.l_h_q_eva = [dict(load_gdeval_res(eval_in, False)) for eval_in in self.l_eval_in]
        self.l_h_qid_e_att = [self._load_e_att(att_in) for att_in in self.l_e_att_in]
        logging.info('eval res and e att res loaded')

    def _load_e_att(self, att_in):
        h_qid_e_att = dict()
        for line in open(att_in):
            h = json.loads(line)
            h_qid_e_att[h['qid']] = h
        return h_qid_e_att

    def _form_one_q(self, qid):
        if qid not in self.l_h_q_eva[0]:
            return ""
        if qid not in self.l_h_qid_e_att[0]:
            return ""
        l_info = [h[qid] for h in self.l_h_qid_e_att]


        query = l_info[0]['query']
        q_pre = qid + '\t' + query
        root_ndcg = self.l_h_q_eva[0][qid][0]
        q_pre += '\t%.4f' % root_ndcg
        key_ndcg = root_ndcg
        if len(self.l_h_q_eva) > 1:
            q_pre += "\t" + "\t".join(['%.4f' % (h_q_eva[qid][0] - root_ndcg)
                                       for h_q_eva in self.l_h_q_eva[1:]])
            key_ndcg = self.l_h_q_eva[1][qid][0] - root_ndcg
        l_qt = query.split()

        l_res_line = []
        for i, loc in enumerate(l_info[0]['sf_ref']):
            sf = ' '.join(l_qt[loc[0]: loc[1]])
            for j in xrange(len(l_info[0]['e_ref'][i])):
                e_id, root_score, name, desp = l_info[0]['e_att_score'][i][j]
                root_score = max(root_score, 0)
                if not name:
                    name = "null"
                if type(root_score) == list:
                    root_score = root_score[0]
                this_line = q_pre + '\t' + '\t'.join([sf, e_id, name, desp]) + '\t%f' % root_score
                for info in l_info[1:]:
                    e_score = info['e_att_score'][i][j][1]
                    if type(e_score) == list:
                        e_score = e_score[0]
                    e_score = max(e_score, 0)
                    this_line += '\t%.4f' % (e_score)
                l_res_line.append(this_line)
            l_res_line.append('\n')
        l_res_line.append('\n\n')
        return key_ndcg, l_res_line

    def process(self):
        logging.info('start aligning eval and e att results')
        out = open(self.out_name, 'w')
        l_key_l_res = []
        for qid in self.l_h_qid_e_att[0].keys():
            key_ndcg, l_lines = self._form_one_q(qid)
            l_key_l_res.append((key_ndcg, l_lines))
            logging.info('q [%s] results get', qid)
        logging.info('sort...')
        l_key_l_res.sort(key=lambda item: item[0])
        for key, l_lines in l_key_l_res:
            print >> out, '\n'.join(l_lines)
        out.close()
        logging.info('finished')


if __name__ == '__main__':
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log
    )
    set_basic_log(logging.INFO)

    if 2 != len(sys.argv):
        print "get e att aligned results for manual analysis"
        PrettyCompEAtt.class_print_help()
        sys.exit(-1)

    ana = PrettyCompEAtt(config=load_py_config(sys.argv[1]))
    ana.process()






