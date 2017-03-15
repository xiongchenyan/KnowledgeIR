"""
convert json style model input (grounding and matchign features)
to numpy ready format
    json, but aligned and qrels feature matrices (list in the disk)

input:
    grounded q json
    matched q-d pairs (match fields ready)
        make sure the features are all full (no missing feature values)
    qrel
    max sf per q (3)
    max e per sf (3)
output:
    one line per pair
        label:
        sf matrix: |spot|*|f dim|
        sf-entity ground: (tensor) |spot||candidate entity||f dim|
        sf_entity qe_d: |spot||candidate entity||f dim|
        all the three's corresponding dimension are aligned
        meta:
            sf: list of loc for sf matrix first dim
            sf-entity matrix: list of sf for sf_entity
            qid
            docno
    full meta:
        sf feature mapping
        sf-e grounding feature mapping
        sf-e matching feature mapping
"""


import json
import sys
import logging
from knowledge4ir.utils import (
    load_trec_labels_dict
)
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)
from knowledge4ir.joint import (
    MATCH_FIELD,
    GROUND_FIELD
)


class ModelInputConvert(Configurable):
    max_spot_per_q = Int(3, help='max spot allowed per q').tag(config=True)
    max_e_per_spot = Int(3, help='top e allowed per q').tag(config=True)
    qrel_in = Unicode(help='qrel in').tag(config=True)
    q_info_in = Unicode(help='q info in, with grounded features').tag(config=True)
    q_d_match_info_in = Unicode(help='matched pairs info in, with matching features'
                                ).tag(config=True)
    out_name = Unicode(help='output prefix').tag(config=True)

    sf_ground_name = Unicode('sf_ground')
    sf_ground_ref = Unicode('sf_ref')
    e_ground_name = Unicode('e_ground')
    e_ground_ref = Unicode('e_ref')
    e_match_name = Unicode('e_match')

    def __init__(self, **kwargs):
        super(ModelInputConvert, self).__init__(**kwargs)
        self.h_qrel = load_trec_labels_dict(self.q_info_in)

        self.h_q_grounding_info_mtx = dict()  # qid -> grounding info mtx
        self.h_sf_grounding_feature_id = dict()
        self.h_e_grounding_feature_id = dict()
        self.h_e_matching_feature_id = dict()

    def _form_q_grounding_info_mtx(self):
        """
        fill self.h_q_grounding_info_mtx and self.h_sf_grounding_feature_id, self.h_e_grounding_feature_id
        :return: self.h_q_grounding_info_mtx and grounding feature id's
        """

        logging.info('starting converting q grounding part')

        for p, line in enumerate(open(self.q_info_in)):
            if not p % 10:
                logging.info('converting [%d] q', p)
            q_info = json.loads(line)
            qid = q_info['qid']
            q_mtx_info = self._assemble_one_q(q_info)
            self.h_q_grounding_info_mtx[qid] = q_mtx_info
        logging.info('q grounding info converted')
        return

    def _assemble_one_q(self, q_info):
        """
        essemble one query
        :param q_info:
        :return: q_mtx_info
        """
        ll_sf_feature = [[] for p in xrange(self.max_spot_per_q)]
        l_spot_loc = []
        lll_sf_e_feature = [[[] for i in xrange(self.max_e_per_spot)]
                            for p in xrange(self.max_spot_per_q)]
        ll_sf_e_id = []

        l_sf_info = q_info[GROUND_FIELD]['query']

        # sf grounding feature
        sf_f_dim = 0
        for i, sf_info in enumerate(l_sf_info[:self.max_spot_per_q]):
            h_f = sf_info['f']
            l_f_score, self.h_sf_grounding_feature_id = self._form_feature_vector(
                h_f, self.h_sf_grounding_feature_id)

            ll_sf_feature[i] = l_f_score
            sf_f_dim = len(l_f_score)
            loc = sf_info['loc']
            l_spot_loc.append(loc)

        # padding
        sf_mtx_shape = [self.max_spot_per_q, sf_f_dim]
        ll_sf_feature = self._padding_mtx(ll_sf_feature,
                                          sf_mtx_shape)

        # e grounding feature tensor
        e_f_dim = 0
        for i, sf_info in enumerate(l_sf_info[:self.max_spot_per_q]):
            l_e_id = []
            for j, e_info in enumerate(sf_info['entities'][:self.max_e_per_spot]):
                h_f = e_info['f']
                l_f_score, self.h_e_grounding_feature_id = self._form_feature_vector(
                    h_f, self.h_e_grounding_feature_id)
                e_f_dim = len(l_f_score)
                e_id = e_info['id']
                lll_sf_e_feature[i][j] = l_f_score
                l_e_id.append(e_id)
            ll_sf_e_id.append(l_e_id)
            for p in xrange(len(sf_info['entities']), self.max_e_per_spot):
                lll_sf_e_feature[i][p] = [0] * e_f_dim

        # padding
        sf_e_tensor_shape = [self.max_spot_per_q, self.max_e_per_spot, e_f_dim]
        lll_sf_e_feature = self._padding_tensor(
            lll_sf_e_feature, sf_e_tensor_shape)

        q_mtx_info = dict()
        q_mtx_info[self.sf_ground_name] = ll_sf_feature
        q_mtx_info[self.sf_ground_ref] = l_spot_loc
        q_mtx_info[self.e_ground_name] = lll_sf_e_feature
        q_mtx_info[self.e_ground_ref] = ll_sf_e_id

        logging.info('q grounding features assembled, sf mtx shape: %s, sf-e tensor shape: %s',
                     json.dumps(sf_mtx_shape), json.dumps(sf_e_tensor_shape))

        return q_mtx_info

    def _assemble_one_pair(self, pair_info, q_grounding_info):
        """
        essemble one pair, update feature id's if new
        features in to first level lists
        meta data to "meta":
        :param pair_info:
        :param q_grounding_info:
        :return:
        """
        converted_mtx_info = dict()
        qid = pair_info['qid']
        docno = pair_info['docno']
        label = self.h_qrel[qid].get(docno, 0)
        logging.info('start assemble par [%s-%s]', qid, docno)

        converted_mtx_info['meta'] = {'qid': qid, 'docno': docno}
        converted_mtx_info['label'] = label
        converted_mtx_info['letor_f'] = [pair_info['base_score']]  # 1 dim ltr feature for now

        # get q's grounding part
        q_mtx_info = q_grounding_info[qid]
        l_spot_loc = q_mtx_info[self.sf_ground_ref]
        ll_sf_e_id = q_mtx_info[self.e_ground_ref]

        h_spot_loc_p = dict(zip(l_spot_loc, range(len(l_spot_loc))))
        l_h_sf_e_p = [dict(zip(l_sf_e_id, range(len(l_sf_e_id))))
                      for l_sf_e_id in ll_sf_e_id]

        logging.info('corresponding q groudning info fetched')

        # form sf-e-feature tensor
        lll_sf_e_match = [[[] for i in xrange(self.max_e_per_spot)]
                          for p in xrange(self.max_spot_per_q)]


        f_dim = 0
        for sf_info in converted_mtx_info[MATCH_FIELD]:
            i = h_spot_loc_p[sf_info['loc']]
            for e_info in sf_info['entities']:
                e_id = e_info['id']
                h_feature = e_info['f']
                j = l_h_sf_e_p[i][e_id]
                l_f_score, self.h_e_matching_feature_id = self._form_feature_vector(
                    h_feature, self.h_e_matching_feature_id
                )
                lll_sf_e_match[i][j] = l_f_score
                f_dim = len(l_f_score)

        # padding
        sf_e_tensor_shape = [self.max_spot_per_q, self.max_e_per_spot, f_dim]
        lll_sf_e_match = self._padding_tensor(
            lll_sf_e_match, sf_e_tensor_shape)

        # put various data into designated locations
        converted_mtx_info['meta'][self.sf_ground_ref] = l_spot_loc
        converted_mtx_info['meta'][self.e_ground_ref] = ll_sf_e_id
        converted_mtx_info[self.sf_ground_name] = q_mtx_info[self.sf_ground_name]
        converted_mtx_info[self.e_ground_name] = q_mtx_info[self.e_ground_name]
        converted_mtx_info[self.e_match_name] = lll_sf_e_match

        logging.info('pair [%s-%s] assembled, matching shape=%s',
                     qid, docno, json.dumps(sf_e_tensor_shape))
        return converted_mtx_info

    def convert(self):
        """

        :return:
        """
        logging.info('start converting input')
        self._form_q_grounding_info_mtx()

        out = open(self.out_name + '.json', 'w')

        for p, line in enumerate(open(self.q_d_match_info_in)):
            if not p % 100:
                logging.info('converted [%d] lines', p)

            pair_info = json.loads(line)
            qid = pair_info['qid']
            q_grounding_info = self.h_q_grounding_info_mtx[qid]
            pair_mtx_info = self._assemble_one_pair(pair_info, q_grounding_info)
            print >> out, json.dumps(pair_mtx_info)

        out.close()
        logging.info('matching pairs converted')

        self._dump_meta()
        logging.info('full grounding and matching data converted')
        return

    def _dump_meta(self):
        out = open(self.out_name + '.meta', 'w')
        h_meta = dict([
            ('sf_grounding_feature', self.h_sf_grounding_feature_id),
            ('e_grounding_feature', self.h_e_grounding_feature_id),
            ('e_matching_feature', self.h_e_matching_feature_id)
        ])
        json.dump(h_meta, out)
        out.close()
        logging.info('meta data (feature name:dim) dumped')
        return

    @classmethod
    def _form_feature_vector(cls, h_feature, h_feature_name):
        if not h_feature_name:
            l_name = h_feature.keys()
            l_name.sort()
            h_feature_name.update(dict(zip([l_name, range(len(l_name))])))

        l_f_scores = [0 for p in len(h_feature)]
        for name, value in h_feature.items():
            assert name in h_feature_name  # assume all feature dicts are extracted fully
            p = h_feature_name[name]
            l_f_scores[p] = value
        return l_f_scores, h_feature_name

    @classmethod
    def _padding_mtx(cls, ll, shape):
        for i in xrange(shape[0]):
            if not ll[i]:
                ll[i] = [0] * shape[-1]

        return ll

    @classmethod
    def _padding_tensor(cls, lll, shape):
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                lll[i][j] = [0] * shape[-1]
        return lll


if __name__ == '__main__':
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log(logging.INFO)

    if 2 != len(sys.argv):
        print "convert readable features into lists"
        print "1 para: config"
        ModelInputConvert.class_print_help()
        sys.exit(-1)

    conf = load_py_config(sys.argv[1])
    converter = ModelInputConvert(config=conf)

    converter.convert()
    
