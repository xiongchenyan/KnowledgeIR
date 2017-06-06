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
    load_trec_labels_dict,
    load_svm_feature,
)
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    Bool
)
from knowledge4ir.joint import (
    MATCH_FIELD,
    GROUND_FIELD
)
from knowledge4ir.joint.att_les import (
    sf_ground_name,
    sf_ground_ref,
    e_ground_name,
    e_ground_ref,
    e_match_name,
    ltr_feature_name
)
import numpy as np
from sklearn.preprocessing import minmax_scale


class ModelInputConvert(Configurable):
    max_spot_per_q = Int(3, help='max spot allowed per q').tag(config=True)
    max_e_per_spot = Int(5, help='top e allowed per q').tag(config=True)
    sf_ground_f_dim = Int(5, help='sf ground feature dimension').tag(config=True)
    e_ground_f_dim = Int(5, help='e ground feature dimension').tag(config=True)
    e_match_f_dim = Int(16, help='e match feature dimension').tag(config=True)
    sf_ground_normalize = Bool(False, help="whether normalize st ground").tag(config=True)
    e_ground_normalize = Bool(False, help="whether normalize e ground").tag(config=True)
    # e_match_normalize = Bool(False, help="whether normalize e match").tag(config=True)
    ltr_f_dim = Int(1, help='ltr feature dimension').tag(config=True)
    ltr_f_in = Unicode(help="pre extracted letor features").tag(config=True)
    qrel_in = Unicode(help='qrel in').tag(config=True)
    q_info_in = Unicode(help='q info in, with grounded features').tag(config=True)
    q_d_match_info_in = Unicode(help='matched pairs info in, with matching features'
                                ).tag(config=True)
    out_name = Unicode(help='output prefix').tag(config=True)

    def __init__(self, **kwargs):
        super(ModelInputConvert, self).__init__(**kwargs)
        self.h_qrel = load_trec_labels_dict(self.qrel_in)

        self.h_q_grounding_info_mtx = dict()  # qid -> grounding info mtx
        self.h_sf_grounding_feature_id = dict()
        self.h_e_grounding_feature_id = dict()
        self.h_e_matching_feature_id = dict()
        self.h_qid_docno_ltr_feature = dict()
        if self.ltr_f_in:
            self._load_svm_ltr_feature()

    def _load_svm_ltr_feature(self):
        logging.info('loading svm ltr feature [%s]', self.ltr_f_in)
        l_svm_data = load_svm_feature(self.ltr_f_in)
        for svm_data in l_svm_data:
            qid, docno, h_feature = svm_data['qid'], svm_data['comment'], svm_data['feature']
            self.h_qid_docno_ltr_feature[qid + '\t' + docno] = h_feature
        logging.info('loaded [%d] pairs of pre extracted ltr feature', len(self.h_qid_docno_ltr_feature))

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
        if self.sf_ground_normalize:
            self._normalize_grounding(sf_ground_name)
        if self.e_ground_normalize:
            self._normalize_grounding(e_ground_name)
        return

    def _normalize_grounding(self, key):
        """
        normalize the one in h_q_grounding_infor_mtx.values, with key -> mtx | tensor
        mtx or tensor are both lists, will convert to a big numpy matrix, normalize, and assign back
        :param key: the field to normalize: sf_ground_name | e_ground_name
        :return:
        """

        l_qid_ts = [(qid, value[key]) for qid, value in self.h_q_grounding_info_mtx.items()]
        l_total_ts = [item[1] for item in l_qid_ts]
        m_total_ts = np.array(l_total_ts)  # the last dimension is the feature

        org_shape = m_total_ts.shape
        v_total_ts = m_total_ts.reshape((-1, org_shape[-1]))
        v_total_ts = np.maximum(v_total_ts, 0)
        v_total_ts = minmax_scale(v_total_ts, axis=0)

        normalized_ts = v_total_ts.reshape(m_total_ts.shape)

        for i in xrange(len(l_qid_ts)):
            qid = l_qid_ts[i][0]
            self.h_q_grounding_info_mtx[qid][key] = normalized_ts[i].tolist()
        logging.info('[%s] normalized', key)
        return

    def _assemble_one_q(self, q_info):
        """
        essemble one query
        :param q_info:
        :return: q_mtx_info
        """
        ll_sf_feature = [[] for _ in xrange(self.max_spot_per_q)]
        l_spot_loc = []
        lll_sf_e_feature = [[[] for __ in xrange(self.max_e_per_spot)]
                            for _ in xrange(self.max_spot_per_q)]
        ll_sf_e_id = []

        l_sf_info = q_info[GROUND_FIELD]['query']

        # sf grounding feature
        for i, sf_info in enumerate(l_sf_info[:self.max_spot_per_q]):
            h_f = sf_info['f']
            l_f_score, self.h_sf_grounding_feature_id = self._form_feature_vector(
                h_f, self.h_sf_grounding_feature_id)

            ll_sf_feature[i] = l_f_score
            loc = tuple(sf_info['loc'])
            l_spot_loc.append(loc)

        # padding
        sf_mtx_shape = [self.max_spot_per_q, self.sf_ground_f_dim]
        ll_sf_feature = self._padding_mtx(ll_sf_feature,
                                          sf_mtx_shape)

        # e grounding feature tensor
        for i, sf_info in enumerate(l_sf_info[:self.max_spot_per_q]):
            l_e_id = []
            for j, e_info in enumerate(sf_info['entities'][:self.max_e_per_spot]):
                h_f = e_info['f']
                l_f_score, self.h_e_grounding_feature_id = self._form_feature_vector(
                    h_f, self.h_e_grounding_feature_id)
                e_id = e_info['id']
                lll_sf_e_feature[i][j] = l_f_score
                l_e_id.append(e_id)
            ll_sf_e_id.append(l_e_id)
            for p in xrange(len(sf_info['entities']), self.max_e_per_spot):
                lll_sf_e_feature[i][p] = [0] * self.e_ground_f_dim

        # padding
        sf_e_tensor_shape = [self.max_spot_per_q, self.max_e_per_spot, self.e_ground_f_dim]
        lll_sf_e_feature = self._padding_tensor(
            lll_sf_e_feature, sf_e_tensor_shape)

        q_mtx_info = dict()
        q_mtx_info[sf_ground_name] = ll_sf_feature
        q_mtx_info[sf_ground_ref] = l_spot_loc
        q_mtx_info[e_ground_name] = lll_sf_e_feature
        q_mtx_info[e_ground_ref] = ll_sf_e_id

        logging.info('q grounding features assembled, sf mtx shape: %s, sf-e tensor shape: %s',
                     json.dumps(sf_mtx_shape), json.dumps(sf_e_tensor_shape))

        return q_mtx_info

    def _assemble_one_pair(self, pair_info, q_grounding_info):
        """
        essemble one pair, update feature id's if new
        features in to first level lists
        meta data to "meta":
        :param pair_info: the match info for this q-d pair
        :param q_grounding_info: the converted grounded query infor for this query
        :return:
        """
        converted_mtx_info = dict()
        qid = pair_info['qid']
        docno = pair_info['docno']
        label = self.h_qrel.get(qid, {}).get(docno, 0)
        logging.info('start assemble par [%s-%s]', qid, docno)

        converted_mtx_info['meta'] = {'qid': qid, 'docno': docno}
        converted_mtx_info['label'] = label
        if not self.h_qid_docno_ltr_feature:
            converted_mtx_info[ltr_feature_name] = [pair_info['base_score']]  # 1 dim ltr feature for now
        else:
            # add in external ltr feature
            h_feature = self.h_qid_docno_ltr_feature.get(qid + '\t' + docno, {})
            if h_feature:
                l_item = h_feature.items()
                l_item.sort(key=lambda item: int(item[0]))
                l_score = [item[1] for item in l_item]
                assert len(l_score) == self.ltr_f_dim
            else:
                l_score = [0 for _ in xrange(self.ltr_f_dim)]
            converted_mtx_info[ltr_feature_name] = l_score

        # get q's grounding part
        l_spot_loc = q_grounding_info[sf_ground_ref]
        ll_sf_e_id = q_grounding_info[e_ground_ref]

        h_spot_loc_p = dict(zip(l_spot_loc, range(len(l_spot_loc))))
        l_h_sf_e_p = [dict(zip(l_sf_e_id, range(len(l_sf_e_id))))
                      for l_sf_e_id in ll_sf_e_id]

        logging.info('corresponding q grounding info fetched')

        # form sf-e-feature tensor
        lll_sf_e_match = [[[] for __ in xrange(self.max_e_per_spot)]
                          for _ in xrange(self.max_spot_per_q)]

        logging.debug('spot loc: %s', json.dumps(l_spot_loc))
        logging.debug('s-e id: %s', json.dumps(ll_sf_e_id, indent=1))

        for sf_info in pair_info[MATCH_FIELD]:
            loc = tuple(sf_info['loc'])
            if loc not in h_spot_loc_p:
                logging.debug('%s not in spot list, perhaps filtered', json.dumps(loc))
                continue
            i = h_spot_loc_p[loc]
            logging.debug('%s i=%d', json.dumps(loc), i)
            for e_info in sf_info['entities']:
                e_id = e_info['id']
                h_feature = e_info['f']
                if e_id not in l_h_sf_e_p[i]:
                    logging.debug('[%s] not in ground (filtered)', e_id)
                    continue
                j = l_h_sf_e_p[i][e_id]
                logging.debug('[%s] j=%d', e_id, j)
                l_f_score, self.h_e_matching_feature_id = self._form_feature_vector(
                    h_feature, self.h_e_matching_feature_id
                )
                lll_sf_e_match[i][j] = l_f_score

        # padding
        sf_e_tensor_shape = [self.max_spot_per_q, self.max_e_per_spot, self.e_match_f_dim]
        lll_sf_e_match = self._padding_tensor(
            lll_sf_e_match, sf_e_tensor_shape)

        # put various data into designated locations
        converted_mtx_info['meta'][sf_ground_ref] = l_spot_loc
        converted_mtx_info['meta'][e_ground_ref] = ll_sf_e_id
        converted_mtx_info[sf_ground_name] = q_grounding_info[sf_ground_name]
        converted_mtx_info[e_ground_name] = q_grounding_info[e_ground_name]
        converted_mtx_info[e_match_name] = lll_sf_e_match

        logging.info('pair [%s-%s] assembled, matching shape=%s',
                     qid, docno, json.dumps(sf_e_tensor_shape))
        return converted_mtx_info

    def convert(self):
        """

        :return:
        """
        logging.info('start converting input')
        self._form_q_grounding_info_mtx()

        out = open(self.out_name, 'w')

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
            h_feature_name.update(dict(zip(l_name, range(len(l_name)))))

        l_f_scores = [0 for _ in xrange(len(h_feature))]
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
                if not lll[i][j]:
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
