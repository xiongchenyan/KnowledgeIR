"""
extract query ana features
"""

from traitlets import (
    Unicode,
    List,
)

from knowledge4ir.deprecated.fusion import QAttFeatureExtractor
from knowledge4ir.utils import (
    load_query_info,
)


class QAttAnaFeatureExtractor(QAttFeatureExtractor):
    feature_name_pre = Unicode('QAna')
    ref_q_info_in = Unicode(help='ref query info').tag(config=True)
    l_tagger = List(Unicode, default_value=['tagme', 'cmns', 'facc'])
    l_feature = List(Unicode,
                     default_value=['Coverage', 'NumOfE', 'QLen',
                                    'RefOverlap',
                                    'Lp',
                                    'Score'])

    def __init__(self, **kwargs):
        super(QAttAnaFeatureExtractor, self).__init__(**kwargs)
        self.h_ref_q_info = {}
        self.s_feature = set(self.l_feature)
        self.h_feature_func = dict(zip(['Coverage', 'NumOfE', 'QLen',
                                   'RefOverlap',
                                   'Lp',
                                   'Score'],
                                  [self._coverage, self._num_of_e,
                                   self._q_len, self._ref_overlap,
                                   self._lp, self._score]
                                  ))
        self._load_data()

    def _load_data(self):
        if self.ref_q_info_in:
            self.h_ref_q_info = load_query_info(self.ref_q_info_in)

    def extract(self, qid, h_info):
        h_feature = {}
        l_ana = self._get_ana(h_info)
        for feature_name in self.l_feature:
            if feature_name in self.h_feature_func:
                h_feature.update(self.h_feature_func[feature_name](qid, h_info, l_ana))
        return h_feature

    def _coverage(self, qid, h_info, l_ana):
        name = self.feature_name_pre + 'Coverage'
        if not l_ana:
            score = 0
            return {name: score}
        l_st_ed = [(int(ana[1]), int(ana[2])) for ana in l_ana]
        l_st_ed.sort(key=lambda item: item[0])
        miss_cnt = l_st_ed[0][0]
        current_ed = l_st_ed[0][1]
        for st, ed in l_st_ed[1:]:
            if st > current_ed:
                miss_cnt += st - ed
            current_ed = max(current_ed, ed)
        q_len = len(h_info['query'])
        miss_cnt += q_len - current_ed
        score = float(miss_cnt) / q_len
        return {name: score}

    def _num_of_e(self, qid, h_info, l_ana):
        return {self.feature_name_pre + 'NumOfE': len(l_ana)}

    def _q_len(self, qid, h_info, l_ana):
        return {self.feature_name_pre + 'QLen': len(h_info['query'].split())}

    def _ref_overlap(self, qid, h_info, l_ana):
        l_ref_ana = []
        s_e = set([ana[0] for ana in l_ana])
        if qid in self.h_ref_q_info:
            this_q_info = self.h_ref_q_info[qid]
            l_ref_ana = self._get_ana(this_q_info)
        s_ref_e = set([ana[0] for ana in l_ref_ana])
        overlap_cnt = len(s_e.intersection(s_ref_e))
        h_feature = dict()
        h_feature[self.feature_name_pre + 'RefOverlap'] = overlap_cnt
        h_feature[self.feature_name_pre + 'RefBoolAnd'] = int(overlap_cnt == len(s_e))
        h_feature[self.feature_name_pre + 'RefBoolOr'] = int(overlap_cnt>0)
        h_feature[self.feature_name_pre + 'RefFrac'] = overlap_cnt / max(1, float(len(s_e)))
        return h_feature

    def _lp(self, qid, h_info, l_ana):
        h_feature = dict()
        l_lp = [0] * (max(len(l_ana), 1))
        if 'tagme' in h_info:
            l_lp = [ana[3]['lp'] for ana in h_info['tagme']['query']]
        if not l_lp:
            l_lp = [0]
        h_feature[self.feature_name_pre + 'MaxLp'] = max(l_lp)
        h_feature[self.feature_name_pre + 'MeanLp'] = sum(l_lp) / max(1, len(l_lp))
        h_feature[self.feature_name_pre + 'MinLp'] = min(l_lp)
        return h_feature

    def _score(self, qid, h_info, l_ana):
        h_feature = dict()
        l_score = [0] * (max(len(l_ana), 1))
        if 'tagme' in h_info:
            l_score = [ana[3]['score'] for ana in h_info['tagme']['query']]
        if not l_score:
            l_score = [0]
        h_feature[self.feature_name_pre + 'MaxScore'] = max(l_score)
        h_feature[self.feature_name_pre + 'MeanScore'] = sum(l_score) / max(1, len(l_score))
        h_feature[self.feature_name_pre + 'MinScore'] = min(l_score)
        return h_feature

    def _get_ana(self, h_info):
        l_ana = []
        for tagger in self.l_tagger:
            if tagger in h_info:
                l_ana.extend(h_info[tagger]['query'])
        return l_ana


