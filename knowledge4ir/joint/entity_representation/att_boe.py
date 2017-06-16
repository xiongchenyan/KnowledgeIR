"""
Constructing boe representation with attention
input:
    spotted documents/query
do:
    form boe
    extract features

output:
    info with fields kept, and boe representation with attention
"""

from knowledge4ir.joint.resource import JointSemanticResource
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    Bool,
    List
)
import logging
import json
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    QUERY_FIELD,
)
from knowledge4ir.joint import SPOT_FIELD


class AttentionBoe(Configurable):
    """
    for (feature) attention based BOE representation
        no soft alignment for now
    """
    l_feature_group = List(Unicode, default_value=['e_vote', 'w_vote', 'sf_ambiguity', 'cmns',],
                           help="attention feature groups: e_vote, w_vote, sf_ambiguity, cmns, coref"
                           ).tag(config=True)
    l_target_field = [QUERY_FIELD] + TARGET_TEXT_FIELDS

    def __init__(self, **kwargs):
        super(AttentionBoe, self).__init__(**kwargs)
        self.resource = None

    def set_resource(self, resource):
        self.resource = resource

    def form_boe(self, h_info):
        """
        for each field in h_info:
            l_e = [{id:, loc:, sf:}]
        :param h_info:
        :return: h_field_boe
        """
        h_field_boe = dict()
        for field in self.l_target_field:
            if field in h_info:
                h_field_boe[field] = self._form_boe_per_field(h_info, field)
        return h_field_boe

    def _form_boe_per_field(self, h_info, field):
        l_ana = h_info.get(SPOT_FIELD, {}).get(field, [])
        l_e = []
        for ana in l_ana:
            sf = ana['surface']
            loc = ana['loc']
            e = ana['entities'][0]
            h = {'surface':sf, 'loc':loc, 'id':e}
            l_e.append(h)
        return l_e

    def extract_att_feature(self, h_info, h_field_boe):
        """
        call feature extractors to extract the target group of features
        :param h_info:
        :param h_field_boe:
        :return: add feature in h_field_boe's each element
        """
        return {}

    def pipe_run(self, in_name, out_name):
        """
        pipe run
            form info from each line in in_name
            form boe and extract att features
            copy target_fields
            dump to out_name, one line per data
        :param in_name:
        :param out_name:
        :return:
        """
        assert self.resource
        logging.info('constructing attention-boe for [%s]', in_name)
        out = open(out_name, 'w')
        for p, line in enumerate(open(in_name)):
            if not p % 100:
                logging.info('constructed [%d] line', p)

            h_info = json.loads(line)
            h_field_boe = self.form_boe(h_info)
            h_field_att_boe = self.extract_att_feature(h_info, h_field_boe)
            for field in self.l_target_field:
                if field in h_info:
                    h_field_att_boe[field] = h_info[field]

            print >> out, json.dumps(h_field_att_boe)
        out.close()
        logging.info('attention boe representations dumped to [%s]', out_name)
        return


class AttBoeMainPara(Configurable):
    spotted_info_in = Unicode(help='spotted query or doc info').tag(config=True)
    out_name = Unicode(help='out name').tag(config=True)


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "config:"
        AttBoeMainPara.class_print_help()
        AttentionBoe.class_print_help()
        JointSemanticResource.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    main_para = AttBoeMainPara(config=conf)
    resource = JointSemanticResource(config=conf)
    boe_constructor = AttentionBoe(config=conf)
    boe_constructor.set_resource(resource)

    boe_constructor.pipe_run(main_para.spotted_info_in, main_para.out_name)




