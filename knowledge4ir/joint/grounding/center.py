"""
the center class to extract features for entities

03/09/2017
currently all features are extracted by one classes.
the logic here is only input-output stream control

"""

import json
import logging
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)
from knowledge4ir.joint.grounding.mem_e import EMemGrounder
from knowledge4ir.joint.grounding import Grounder
from knowledge4ir.joint.resource import JointSemanticResource
from knowledge4ir.utils import TARGET_TEXT_FIELDS
from copy import deepcopy


class GroundCenter(Configurable):
    in_name = Unicode(help='q info or doc info spotted json file to ground'
                      ).tag(config=True)
    out_name = Unicode(help='output file name').tag(config=True)
    grounder_name = Unicode('base',
                            help='the grounder to use, default base, choice: base|e_mem'
                            ).tag(config=True)

    def __init__(self, **kwargs):
        super(GroundCenter, self).__init__(**kwargs)
        if self.grounder_name == 'base':
            self.grounder = EMemGrounder(**kwargs)
        else:
            self.grounder = Grounder(**kwargs)
        self.resource = JointSemanticResource(**kwargs)
        self.grounder.set_resource(self.resource)

    @classmethod
    def class_print_help(cls, inst=None):
        super(GroundCenter, cls).class_print_help(inst)
        JointSemanticResource.class_print_help(inst)
        Grounder.class_print_help(inst)

    def ground(self):
        """
        main api
        ground self.in_name to self.out_name
        :return:
        """
        logging.info('start grounding [%s]', self.in_name)
        out = open(self.out_name, 'w')
        l_h_grounded_info = []
        for p, line in enumerate(open(self.in_name)):
            if not (p % 10):
                logging.info('grounded [%d] lines', p)

            h_info = json.loads(line)
            h_grounded_info = self.grounder.extract(h_info)
            l_h_grounded_info.append(h_grounded_info)

        logging.info('all grounded, add feature place holders')
        l_h_grounded_info = self._add_zero_f(l_h_grounded_info)
        for h_grounded_info in l_h_grounded_info:
            print >> out, json.dumps(h_grounded_info)

        logging.info('finished, results at [%s]', self.out_name)
        return

    def _add_zero_f(self, l_h_info):
        """
        add 0 values to missing features
        :param l_h_info: each one is a grounded info
        :return:
        """
        h_sf_f = dict()
        h_e_f = dict()

        for h_info in l_h_info:
            for field in TARGET_TEXT_FIELDS + ['query']:
                if field not in h_info['ground']:
                    continue
                l_sf_ground = h_info['ground'][field]
                for h_sf in l_sf_ground:
                    this_sf_f = h_sf['f']
                    h_sf_f.update(this_sf_f)
                    l_e = h_sf['entities']
                    for h_e in l_e:
                        this_e_f = h_e['f']
                        h_e_f.update(this_e_f)

        for key in h_sf_f:
            h_sf_f[key] = 0
        for key in h_e_f:
            h_e_f[key] = 0

        l_res = []
        for h_info in l_h_info:
            for field in TARGET_TEXT_FIELDS + ['query']:
                if field not in h_info['ground']:
                    continue
                l_sf_ground = h_info['ground'][field]
                for h_sf in l_sf_ground:
                    new_f = deepcopy(h_sf_f)
                    new_f.update(h_sf['f'])
                    h_sf['f'] = new_f
                    l_e = h_sf['entities']
                    for h_e in l_e:
                        new_e_f = deepcopy(h_e_f)
                        new_e_f.update(h_e['f'])
                        h_e['f'] = new_e_f
            l_res.append(h_info)
        return l_res


if __name__ == '__main__':
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    import sys

    set_basic_log(logging.DEBUG)
    if 2 != len(sys.argv):
        print "I ground spotted entities"
        print "1 para: config"
        GroundCenter.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    center = GroundCenter(config=conf)
    center.ground()

