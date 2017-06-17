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
    List
)
import logging
import json
from knowledge4ir.utils import (
    TARGET_TEXT_FIELDS,
    QUERY_FIELD,
)
from knowledge4ir.joint.utils import (
    form_boe_per_field,
    surface_ambiguity_feature,
    surface_lp,
    word_embedding_vote,
    entity_embedding_vote,
    uw_word_embedding_vote,
    cmns_feature,
)


class AttentionBoe(Configurable):
    """
    for (feature) attention based BOE representation
        no soft alignment for now
    """
    l_feature_group = List(Unicode, default_value=['e_vote', 'w_vote', 'sf_ambiguity', 'cmns'],
                           help="attention feature groups: e_vote, w_vote,"
                                "uw_w_vote, sf_ambiguity, cmns"
                           ).tag(config=True)
    l_target_field = [QUERY_FIELD] + TARGET_TEXT_FIELDS

    s_supported_feature = {'e_vote', 'w_vote', 'sf_ambiguity', 'cmns', 'uw_w_vote'}

    def __init__(self, **kwargs):
        super(AttentionBoe, self).__init__(**kwargs)
        self.resource = None
        for f_group in self.l_feature_group:
            assert f_group in self.s_supported_feature

    def set_resource(self, resource):
        self.resource = resource
        if 'e_vote' in self.l_feature_group:
            assert self.resource.entity_embedding
        if ('w_vote' in self.l_feature_group) | ('uw_w_vote' in self.l_feature_group):
            assert self.resource.embedding
        # if 'sf_ambiguity' in self.l_feature_group:
        #     assert self.resource.h_surface_stat

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
                h_field_boe[field] = form_boe_per_field(h_info, field)
        return h_field_boe

    def extract_att_feature(self, h_info, h_field_boe):
        """
        call feature extractors to extract the target group of features
        :param h_info:
        :param h_field_boe:
        :return: add feature in h_field_boe's each element
        """

        for field, l_e_info in h_field_boe.items():
            for p in xrange(len(l_e_info)):
                h_feature = self._extract_per_e_att_feature(l_e_info[p], h_info, field)
                h_field_boe[field][p]['feature'] = h_feature
        return h_field_boe

    def _extract_per_e_att_feature(self, e_info, h_info, field):
        h_feature = dict()
        e_id = e_info['id']
        sf = e_info['surface']
        loc = e_info['loc']
        logging.debug('extracting [%s] [%s]', json.dumps(e_id), json.dumps(loc))
        if 'e_vote' in self.l_feature_group:
            h_feature.update(entity_embedding_vote(e_id, h_info, field, self.resource))
        if 'w_vote' in self.l_feature_group:
            h_feature.update(word_embedding_vote(e_id, h_info, field, self.resource))
        if 'uw_w_vote' in self.l_feature_group:
            h_feature.update(uw_word_embedding_vote(e_id, h_info, field, loc, self.resource))
        if 'sf_ambiguity' in self.l_feature_group:
            h_feature.update(surface_ambiguity_feature(e_id, h_info, field))
            # h_feature.update(surface_lp(sf, self.resource))
        if 'cmns' in self.l_feature_group:
            h_feature.update(cmns_feature(e_id, h_info, field))

        return h_feature

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
        logging.info('constructing attention-boe for [%s], to [%s]', in_name, out_name)
        out = open(out_name, 'w')
        for p, line in enumerate(open(in_name)):
            if not p % 10:
                logging.info('constructed [%d] line', p)
            h_info = json.loads(line)
            logging.info('constructing for [%s]', h_info['docno'])
            h_field_boe = self.form_boe(h_info)
            h_field_att_boe = self.extract_att_feature(h_info, h_field_boe)
            for key in h_field_att_boe.keys():
                logging.info('[%s][%d] entities', key, len(h_field_att_boe[key]))
            h_res = {'boe': h_field_att_boe}
            for field in self.l_target_field:
                if field in h_info:
                    h_res[field] = h_info[field]

            print >> out, json.dumps(h_res)
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
    if 2 > len(sys.argv):
        print "config + spot in (opt) + out name (opt):"
        AttBoeMainPara.class_print_help()
        AttentionBoe.class_print_help()
        JointSemanticResource.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    main_para = AttBoeMainPara(config=conf)
    semantic_resource = JointSemanticResource(config=conf)
    boe_constructor = AttentionBoe(config=conf)
    boe_constructor.set_resource(semantic_resource)
    if len(sys.argv) < 4:
        boe_constructor.pipe_run(main_para.spotted_info_in, main_para.out_name)
    else:
        boe_constructor.pipe_run(sys.argv[2], sys.argv[3])






