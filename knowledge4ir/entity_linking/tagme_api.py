"""
tagme API
"""
import sys
import logging
import json
import time
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)
import urllib
import urllib2

TagMe_Key = 'cad23c26-6f1f-4164-a62e-fc5107c031ab-843339462'
TagMe_URL = 'https://tagme.d4science.org/tagme/tag'


class TagMeAPILinker(Configurable):
    wiki_fb_dict = Unicode(help='wiki fb alignment').tag(config=True)
    
    def __init__(self, **kwargs):
        super(TagMeAPILinker, self).__init__(**kwargs)
        logging.info('load wiki 2 fb dict...')
        self.h_wiki_id_fb = dict([line.split('\t')[:2] for line in open(self.wiki_fb_dict)])
        logging.info('wiki 2 fb dict loaded')

    def link(self, text):
        paras = {'text': text, 'gcube-token':TagMe_Key}
        data = urllib.urlencode(paras)
        req = urllib2.Request(TagMe_URL, data)
        response = urllib2.urlopen(req)
        res_text = response.read()
        h_res = json.loads(res_text)
        if 'annotations' not in h_res:
            logging.warn('tag me [%s] failed', text)
            return []
        l_tagme_ana = h_res['annotations']
        l_ana = []
        for tag in l_tagme_ana:
            name = tag['title']
            wid = tag['id']
            st = tag['start']
            ed = tag['end']
            rho = tag['rho']
            lp = tag['link_probability']
            if wid in self.h_wiki_id_fb:
                fid = self.h_wiki_id_fb[wid]
            else:
                logging.warn('wid [%s] not in wiki id dict', wid)
                continue
            ana = [fid, st, ed, {'lp': lp, 'score': rho}, name]
            l_ana.append(ana)
        return l_ana






