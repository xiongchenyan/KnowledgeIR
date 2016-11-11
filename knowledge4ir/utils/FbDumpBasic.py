'''
Created on Feb 7, 2014
basic operations on Freebase dump data
@author: cx
'''

'''
Oct 27, 2014
Added get multiple alias func
Added get wiki url func
and reconstructed them as a class
'''

'''
Aug 31, 2014,
to not using Google Freebase API (might be depreciated very soon)
    support form FbObj from dump
    support fetch wiki en id, in order to match given wikipedia id
    
'''

'''
Sep 27 2016
refactored for current code base
'''

import json
from traitlets.config import Configurable
import logging


TYPE_EDGE = "<http://rdf.freebase.com/ns/type.object.type>"
DESP_EDGE = "<http://rdf.freebase.com/ns/common.topic.description>"
NAME_EDGE = "<http://www.w3.org/2000/01/rdf-schema#label>"
ALIAS_EDGE = "<http://rdf.freebase.com/ns/common.topic.alias>"
NOTABLE_EDGE = "<http://rdf.freebase.com/ns/common.topic.notable_types>"
INSTANCE_EDGE = "<http://rdf.freebase.com/ns/type.type.instance>"
WIKIURL_EDGE = "<http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage>"
WIKIENID_EDGE = '<http://rdf.freebase.com/key/wikipedia.en_id>'
l_WIKIURL_EDGE = ["<http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage>","<http://rdf.freebase.com/ns/common.topic.topical_webpage>"]


class FbDumpParser(Configurable):

    @staticmethod
    def get_obj_id(l_v_col):
        if not l_v_col:
            logging.warn('put in an empty vcol to parser')
            return ""
        return FbDumpParser.get_id_for_col(l_v_col[0][0])
    
    @staticmethod
    def discard_prefix(col):
        if len(col) < 2:
            return col
        if (col[0] != '<') | (col[len(col) - 1] !=">"):
            return col    
        mid = col.strip("<").strip(">")
        v_col = mid.split("/")
        target = v_col[-1]
        return '/' + target.replace('.', '/')
    
    @staticmethod
    def get_id_for_col(col):
        logging.info('getting id from [%s]', col)
        target = FbDumpParser.discard_prefix(col)
        if len(target) < 2:
            return ""
        if (target[:len('/m/')] == "/m/") | (target[:len('/en/')]=='/en/'):
            return target
        return ""
    
    @staticmethod
    def fetch_targets_with_edge(l_v_col, edge):
        '''
        fetch col with edge (obj edge col)
        '''
        lTar = []
        for vCol in l_v_col:
            if vCol[1] == edge:
                lTar.append(vCol[2])

        return lTar
    
    @staticmethod
    def fetch_target_string_with_edge(l_v_col, edge):
        '''
        same, but only look for english strings
        '''
        l_tar = FbDumpParser.fetch_targets_with_edge(l_v_col, edge)
#         print 'curent obj:%s' %(json.dumps(lvCol))
#         print 'edge [%s] get targets [%s]' %(Edge,json.dumps(lTar))
        l_str = []
        for tar in l_tar:
            if not FbDumpParser.is_string(tar):
                continue
            text,tag = FbDumpParser.seg_language_tag(tar)
            if (tag == "") | (tag == 'en'):
                l_str.append(text)
#         print 'get text [%s]' %(json.dumps(lStr))
        return l_str

    def get_name(self, l_v_col):
        l_str = self.fetch_target_string_with_edge(l_v_col, NAME_EDGE)
        if not l_str:
            return ""
        return l_str[0]
    
    def get_alias(self, l_v_col):
        return self.fetch_target_string_with_edge(l_v_col, ALIAS_EDGE)
    
    def get_desp(self, l_v_col):
        return '\n'.join(self.fetch_target_string_with_edge(l_v_col, DESP_EDGE))
    
    def get_wiki_id(self, l_v_col):
        l_wiki_id = self.fetch_target_string_with_edge(l_v_col, WIKIENID_EDGE)
        if not l_wiki_id:
            return ""
        return l_wiki_id[0]
    
    def get_neighbor(self, l_v_col):
        l_neighbor = []
        for vCol in l_v_col:
            neighbor_id = self.get_id_for_col(vCol[2])
            if "" != neighbor_id:
                neighbor_edge = self.discard_prefix(vCol[1])
                l_neighbor.append([neighbor_edge,neighbor_id])
        return l_neighbor
    
    def get_wiki_url(self, l_v_col):
        l_wiki_url = []
        for edge in l_WIKIURL_EDGE:
            l_tar = self.fetch_targets_with_edge(l_v_col, edge)
#             if [] != lTar:
#                 print 'wiki target %s' %(json.dumps(lTar))
            
            for tar in l_tar:
                if 'http' not in tar:
                    continue
                if 'en.wikipedia' not in tar:
                    continue
                l_wiki_url.append(tar.strip('<').strip('>'))
        return l_wiki_url
    
    def get_type(self, l_v_col):
        lTar = self.fetch_targets_with_edge(l_v_col, TYPE_EDGE)
        lType = []
        for tar in lTar:
            Type = self.discard_prefix(tar)
#             if '/common' == Type[:len('/common')]:
#                 continue          
            lType.append(Type)
        return lType
    
    def get_notable(self, l_v_col):
        l_tar = self.fetch_targets_with_edge(l_v_col, NOTABLE_EDGE)
        if not l_tar:
            return ""
        return self.discard_prefix(l_tar[0])
        
    @staticmethod
    def is_string(s):
        if s[0] != '\"':
            return False
        if s[-1] == '\"':
            return True
        v_col = s.split('@')
        if v_col[0][-1] == '\"':
            return True
        return False
    
    @staticmethod     
    def seg_language_tag(s):
        v_col = s.split("@")
        lang = ""
        text = v_col[0].strip('"')
        if len(v_col) >= 2:
            lang = v_col[1]
        return text,lang
    


