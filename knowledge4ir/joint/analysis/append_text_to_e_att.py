"""
append texts to e attention results

add query
add e name

input:
    e att results of attention_les
    e text dict
    qid \t query

output:
    add re-able things to e att res
"""

import json
import sys
from traitlets.config import Configurable
from traitlets import (
    Unicode
)
from knowledge4ir.utils import load_qid_query
from knowledge4ir.joint.resource import JointSemanticResource
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


class AppendText(Configurable):
    in_name = Unicode(help="e att res in").tag(config=True)
    q_in = Unicode(help='qid query in').tag(config=True)
    out_name = Unicode(help='out name').tag(config=True)

    def __init__(self, **kwargs):
        super(AppendText, self).__init__(**kwargs)
        self.resource = JointSemanticResource(**kwargs)
        self.h_qid_query = dict(load_qid_query(self.q_in))
        self.h_e_text = self.resource.h_entity_fields

    @classmethod
    def class_print_help(cls, inst=None):
        super(AppendText, cls).class_print_help(inst)
        JointSemanticResource.class_print_help(inst)

    def _add_one_data(self, h):
        query = self.h_qid_query[h['qid']]
        h['query'] = query
        l_att_score = h['e_att_score']
        for i in xrange(len(l_att_score)):
            for j in xrange(len(l_att_score[i])):
                e_id = l_att_score[i][j][0]
                name = self.h_e_text.get(e_id, {}).get('name')
                desp = self.h_e_text.get(e_id, {}).get('desp')
                desp = ' '.join(desp.split())
                l_att_score[i][j].extend([name, desp])
        h['e_att_score'] = l_att_score
        return h

    def process(self):
        out = open(self.out_name, 'w')

        for line in open(self.in_name):
            h = json.loads(line)
            h = self._add_one_data(h)
            print >> out, json.dumps(h)

        out.close()
        print "finished"


if __name__ == '__main__':
    from knowledge4ir.utils import load_py_config
    if 2 != len(sys.argv):
        print " add text to e att"
        print "1 para: config"
        AppendText.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    apper = AppendText(config=conf)
    apper.process()





