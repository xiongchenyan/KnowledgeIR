"""
the pipe line running center


load q info
load d info
load resource


for each line in TREC ranking
    do feature extraction
    output features, packed in json
        qid:
        docno:
        base score:
        vector for each spot q e -> d (les, first target
        matrix for each q and d spot: q e -> d e (ESR, TODO
"""


import logging
import json
import sys
from knowledge4ir.joint.resource import JointSemanticResource
from knowledge4ir.joint.matching import QeDTextMatchFeatureExtractor
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Set,
)
from knowledge4ir.utils import (
    load_trec_ranking_with_score
)
from knowledge4ir.joint import (
    MATCH_FIELD
)


class MatchCenter(Configurable):
    s_feature_group = Set(Unicode, default_value={'qe_d'},
                          help='match feature groups: {"qe_d"}'
                          ).tag(config=True)
    q_info_in = Unicode('location of query info (grounded)').tag(config=True)
    d_info_in = Unicode('location of doc info').tag(config=True)
    h_group_mapping = {"qe_d": QeDTextMatchFeatureExtractor}

    def __init__(self, **kwargs):
        super(MatchCenter, self).__init__(**kwargs)
        self.resource = JointSemanticResource(**kwargs)
        self.h_q_info = dict()
        self.h_d_info = dict()
        self._load_qd_info()

        self.l_feature_extractor = []
        self._init_feature_extractors(**kwargs)

    @classmethod
    def class_print_help(cls, inst=None):
        super(MatchCenter, cls).class_print_help(inst)
        JointSemanticResource.class_print_help(inst)
        for name, extractor_class in cls.h_group_mapping.items():
            print 'feature group:[%s]' % name
            extractor_class.class_print_help(inst)

    def _load_qd_info(self):
        """
        load query and doc infor from config traits
        :return:
        """
        logging.info('start loading q info [%s]', self.q_info_in)
        self.h_q_info = self._load_info(self.q_info_in, key='qid')
        logging.info('[%d] q info loaded', len(self.h_q_info))

        logging.info('start loading d info [%s]', self.d_info_in)
        self.h_d_info = self._load_info(self.d_info_in, key='docno')
        logging.info('[%d] d info loaded', len(self.h_d_info))
        return

    @classmethod
    def _load_info(cls, info_in, key='qid'):
        h_info = dict()

        for line in open(info_in):
            h_data = json.loads(line)
            assert key in h_data
            key_id = h_data[key]
            h_info[key_id] = h_data
        return h_info

    def _init_feature_extractors(self, **kwargs):

        for feature_group in self.s_feature_group:
            assert feature_group in self.h_group_mapping

        for feature_group in self.s_feature_group:
            self.l_feature_extractor.append(self.h_group_mapping[feature_group](**kwargs))
            logging.info('feature group [%s] added', feature_group)
        logging.info('total [%d] feature extractor initialized', len(self.l_feature_extractor))
        return

    def pipe_extract(self, trec_rank_in, out_name):
        """
        the main pipe run
        :param trec_rank_in: trec rank format input candidate q-document pairs to extract features
        :param out_name: the extracted matching features, one json per line
        :return:
        """

        l_q_ranking = load_trec_ranking_with_score(trec_rank_in)
        out = open(out_name, 'w')
        for q, ranking in l_q_ranking:
            logging.info('start extracting for [%s]', q)
            q_info = self.h_q_info[q]
            for docno, score in ranking:
                d_info = self.h_d_info.get(docno, {})
                h_matched_feature = dict()
                for extractor in self.l_feature_extractor:
                    h_this_matched_feature = extractor.extract(
                        q_info, d_info, self.resource)
                    self._mul_update(h_matched_feature, h_this_matched_feature)
                print >> out, json.dumps(h_matched_feature)
                logging.info('[%s-%s] match feature extracted', q, docno)
            logging.info('q [%s] match features extracted', q)
        logging.info('ranking pairs [%s] matching features extracted to [%s]',
                     trec_rank_in, out_name)
        return

    def _mul_update(self, h_total, h_this_one):
        """
        add the features in h_this_one to h_total
        :param h_total: MATCH_FIELD->[{sf _dict},] sf_dict = {'surface':, 'entities':{'id',match_f:''}}
        :param h_this_one:
        :return:
        """

        l_total_matched = h_total[MATCH_FIELD]
        l_this_matched = h_this_one[MATCH_FIELD]
        h_this_sf_matched = dict([(h_sf['loc'], h_sf) for h_sf in l_this_matched])

        for i in xrange(len(l_total_matched)):
            l_total_matched[i] = self._update_per_sf(
                l_total_matched[i], h_this_sf_matched.get(l_total_matched[i]['loc']))
        h_total[MATCH_FIELD] = l_total_matched
        return

    def _update_per_sf(self, h_sf_matched, h_to_add_sf):
        assert h_sf_matched['loc'] == h_to_add_sf['loc']
        l_entities_matched = h_sf_matched['entities']
        l_to_add_entities = h_to_add_sf['entities']
        h_to_add_entities = dict([(h['id'], h) for h in l_to_add_entities])

        for p in xrange(len(l_entities_matched)):
            to_add = h_to_add_entities[l_to_add_entities['id']]
            l_entities_matched['f'].update(to_add['f'])
        h_sf_matched['entities'] = l_entities_matched
        return h_sf_matched


class MatchMain(Configurable):
    trec_rank_in = Unicode(help="trec rank in").tag(config=True)
    feature_out = Unicode(help='extract feature out').tag(config=True)


if __name__ == '__main__':
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    set_basic_log(logging.DEBUG)
    if 2 != len(sys.argv):
        print "I extract matching features"
        print "1 para: config"
        MatchCenter.class_print_help()
        MatchMain.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])

    main_para = MatchMain(config=conf)
    extractor = MatchCenter(config=conf)

    extractor.pipe_extract(main_para.trec_rank_in, main_para.feature_out)
