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
from knowledge4ir.joint.grounding import Grounder
from knowledge4ir.joint.resource import JointSemanticResource


class GroundFeatureCenter(Configurable):
    in_name = Unicode(help='q info or doc info spotted json file to ground'
                      ).tag(config=True)
    out_name = Unicode(help='output file name').tag(config=True)

    def __init__(self, **kwargs):
        super(GroundFeatureCenter, self).__init__(**kwargs)
        self.grounder = Grounder(**kwargs)
        self.resource = JointSemanticResource(**kwargs)
        self.grounder.set_resource(self.resource)

    @classmethod
    def class_print_help(cls, inst=None):
        super(GroundFeatureCenter, cls).class_print_help(inst)
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
        for p, line in enumerate(open(self.in_name)):
            if not (p % 10):
                logging.info('grounded [%d] lines', p)

            h_info = json.loads(line)
            h_grounded_info = self.grounder.extract(h_info)
            print >> out, json.dumps(h_grounded_info)

        logging.info('finished, results at [%s]', self.out_name)
        return


if __name__ == '__main__':
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    import sys

    set_basic_log(logging.INFO)
    if 2 != len(sys.argv):
        print "I ground spotted entities"
        print "1 para: config"
        GroundFeatureCenter.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    center = GroundFeatureCenter(config=conf)
    center.ground()

