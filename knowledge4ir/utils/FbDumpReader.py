"""
Created on Apr 21, 2014
read freebase dump
once an object's all triples
cut off at a limited amount of lines
@author: cx
"""


import gzip
import json
import sys
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List,
    Bool,
)
import logging


reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


class KeyFileReader(Configurable):
    is_gzip = Bool(False, help='spot is gzip or not').tag(config=True)
    l_key_inx = List(Int, default_value=[0], help='key columns p').tag(config=True)
    splitter = Unicode('\t', help="spliter").tag(config=True)
    max_line_per_key = Int(100000, help='max line per key').tag(config=True)

    def __init__(self, **kwargs):
        super(KeyFileReader, self).__init__(**kwargs)

    def generate_key(self, v_col):
        key = ""
        for i in self.l_key_inx:
            key += v_col[i] + self.splitter
        return key.strip(self.splitter)

    def read(self, in_name):
        """

        :param in_name: the gz of Freebase dump
        :return: yield triples lv_col for an entity a time.
            lv_col = [[head, predicate, tail], ...]
        """
        lv_col = []  # one object's all triples
        current_key = None
        if self.is_gzip:
            in_file = gzip.open(in_name, 'r')
        else:
            in_file = open(in_name, 'r')
        for line in in_file:
            try:
                v_col = line.strip().split(self.splitter)
            except UnicodeDecodeError:
                continue
            if not v_col:
                continue
            # print "read vcol %s" % json.dumps(v_col)
            this_key = self.generate_key(v_col)

            if not current_key:
                current_key = this_key
                logging.debug("start with %s", current_key)
            if this_key != current_key:
                yield lv_col
                lv_col = []
                current_key = this_key
                logging.debug("get %s", current_key)
            if len(lv_col) < self.max_line_per_key:
                lv_col.append(v_col)
                # print "cuurent total %s" % json.dumps(lv_col)
        if lv_col:
            yield lv_col


class FbDumpReader(KeyFileReader):
    is_gzip = Bool(True).tag(config=True)
    
    def read(self, in_name):
        for lv_col in super(FbDumpReader, self).read(in_name):
            # logging.info("get vcols: %s", json.dumps(lv_col))
            lv_col = [v_col for v_col in lv_col if len(v_col) >= 3]
            yield lv_col


if __name__ == '__main__':
    # unit test
    from knowledge4ir.utils.freebase.FbDumpBasic import FbDumpParser
    if 3 != len(sys.argv):
        print "unit test reader and FbDumpParser"
        print "2 para: dump gzip + output"
        sys.exit()

    out = open(sys.argv[2], 'w')
    reader = FbDumpReader()
    reader.is_gzip = True
    parser = FbDumpParser()
    m_cnt = 0
    for cnt, lv_col in enumerate(reader.read(sys.argv[1])):
        # print json.dumps(lv_col, indent=1)
        name = parser.get_name(lv_col)
        mid = parser.get_obj_id(lv_col)
        desp = parser.get_desp(lv_col)
        if not mid.startswith('/m/'):
            continue
        print >> out, mid + "\t" + name + '\t' + desp
        m_cnt += 1
        if m_cnt > 1000:
            break

    out.close()
