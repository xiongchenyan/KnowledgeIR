"""
annotate a q infor or a doc info
input:
    entity linker type
    entity linker's config data
    to annotate field
    to annotated data
output:
    append an entity link field to the info
        "linker's name": [[eid, st, ed, name, score],]

"""

from traitlets.config import Configurable
from knowledge4ir.entity_linking.cmns import CommonEntityLinker
from knowledge4ir.entity_linking.tagme_api import TagMeAPILinker
from traitlets import (
    Unicode,
    Int,
    List,
)
from knowledge4ir.utils import TARGET_TEXT_FIELDS
import json
import logging
import sys
import time

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


class Annotator(Configurable):
    linker_type = Unicode('cmns', help='type of linker:cmns|tagme').tag(config=True)
    target_fields = List(Unicode, default_value=TARGET_TEXT_FIELDS).tag(config=True)
    in_name = Unicode(help='input').tag(config=True)
    out_name = Unicode(help='output').tag(config=True)
    
    def __init__(self, **kwargs):
        super(Annotator, self).__init__(**kwargs)
        if self.linker_type == 'cmns':
            self.linker = CommonEntityLinker(**kwargs)
        elif self.linker_type == 'tagme':
            self.linker = TagMeAPILinker(**kwargs)
        else:
            raise NotImplementedError

    @classmethod
    def class_print_help(cls, inst=None):
        super(Annotator, cls).class_print_help(inst)
        CommonEntityLinker.class_print_help(inst)

    def annotate(self, in_name=None, out_name=None):
        if not in_name:
            in_name = self.in_name
        if not out_name:
            out_name = self.out_name
        logging.info('start annotating [%s]', in_name)
        out = open(out_name, 'w')
        api_cnt = 0
        for line_cnt, line in enumerate(open(in_name)):
            d_id, d_info = line.split('\t')
            d_info = json.loads(d_info)
            d_info[self.linker_type] = {}
            for field in self.target_fields:
                if field in d_info:
                    l_ana, __ = self.linker.link(d_info[field])
                    api_cnt += 1
                    if not api_cnt % 5:
                        if self.linker_type == 'tagme':
                            time.sleep(1)
                    d_info[self.linker_type][field] = l_ana
            print >> out, d_id + '\t' + json.dumps(d_info)

            if not line_cnt % 100:
                logging.info('annotated [%d] data', line_cnt)
        out.close()
        logging.info('finished [%d] data annotation', line_cnt + 1)


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    from traitlets.config.loader import PyFileConfigLoader
    set_basic_log()
    if 2 != len(sys.argv):
        print "1 para: config"
        Annotator.class_print_help()
        sys.exit(-1)

    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    annotator = Annotator(config=conf)
    annotator.annotate()
    




