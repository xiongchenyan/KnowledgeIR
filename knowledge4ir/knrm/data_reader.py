"""
data i/o
input:
    trec rank candidate docs
    qrels
    q info (tensor format)
    doc info (tensor format)
output:
    pairwise x, y (for target q id)
    pointwise x, y (for target q id)
"""

from knowledge4ir.knrm import (
    aux_pre,
    q_in_name,
    d_in_name,
    ltr_feature_name,
    q_att_name,
    d_att_name,
    q_len,
    l_field_len,
)
from knowledge4ir.utils import (
    load_trec_labels_dict,
    load_trec_ranking_with_score,
    load_json_info,
    TARGET_TEXT_FIELDS
)
import json
import logging
import numpy as np
import os


def padding(boe, max_len):
    while len(boe) < max_len:
        boe.append(0)
    return boe


def padding_2d(ll, max_len):
    n = len(ll[0])
    while len(ll) < max_len:
        ll.append([0] * n)
    return ll


def dynamic_load(trec, qrel, q_info, doc_info):
    if type(trec) == str:
        l_q_rank = load_trec_ranking_with_score(trec)
    else:
        l_q_rank = trec
    if type(qrel) == str:
        h_qrel = load_trec_labels_dict(qrel)
    else:
        h_qrel = qrel
    if type(q_info) == str:
        h_q_info = load_json_info(q_info, 'qid')
    else:
        h_q_info = q_info
    if type(doc_info) == str:
        h_doc_info = load_json_info(doc_info, 'docno')
    else:
        h_doc_info = doc_info
    return l_q_rank, h_qrel, h_q_info, h_doc_info


def pointwise_reader(trec_in, qrel_in, q_info_in, doc_info_in, s_qid=None, with_att=False):
    logging.info('start read pointwise')
    l_q_rank, h_qrel, h_q_info, h_doc_info = dynamic_load(trec_in, qrel_in, q_info_in, doc_info_in)
    logging.info('input data loaded')
    l_label = []
    l_q_in = []
    l_q_att = []
    l_ltr = []
    ll_doc_field = [[] for __ in TARGET_TEXT_FIELDS]
    ll_doc_att = [[] for __ in TARGET_TEXT_FIELDS]
    l_qid = []
    l_docno = []

    for q, rank in l_q_rank:
        if s_qid is not None:
            if q not in s_qid:
                continue
        logging.info('constructing for q [%s]', q)
        q_boe = h_q_info[q]['query']['boe']
        q_boe = padding(q_boe, q_len)
        q_att = h_q_info[q]['query']['att_mtx']
        for docno, score in rank:
            if docno not in h_doc_info:
                continue
            l_ltr.append([score])
            label = h_qrel.get(q, {}).get(docno, 0)
            l_label.append(label)
            l_q_in.append(q_boe)
            doc_info = h_doc_info[docno]
            l_qid.append(q)
            l_docno.append(docno)

            for p in xrange(len(TARGET_TEXT_FIELDS)):
                field = TARGET_TEXT_FIELDS[p]
                ll_doc_field[p].append(padding(doc_info[field]['boe'], l_field_len[p]))

            if with_att:
                l_q_att.append(q_att)
                for p in xrange(len(TARGET_TEXT_FIELDS)):
                    field = TARGET_TEXT_FIELDS[p]
                    ll_doc_att[p].append(padding_2d(doc_info[field]['att_mtx'], l_field_len[p]))
    if with_att:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field, l_q_att, ll_doc_att)
    else:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field)
    x['qid'] = np.array(l_qid)
    x['docno'] = np.array(l_docno)
    logging.info('pointwise data constructed [%d] q, [%d] doc', len(l_q_in), len(l_label))
    return x, y


def pairwise_reader(trec_in, qrel_in, q_info_in, doc_info_in, s_qid=None, with_att=False):
    logging.info('start read pairwise')
    l_q_rank, h_qrel, h_q_info, h_doc_info = dynamic_load(trec_in, qrel_in, q_info_in, doc_info_in)
    logging.info('input data loaded')
    l_label = []
    l_q_in = []
    l_q_att = []
    l_ltr = []
    l_aux_ltr = []
    ll_doc_field = [[] for __ in TARGET_TEXT_FIELDS]
    ll_doc_att = [[] for __ in TARGET_TEXT_FIELDS]
    ll_aux_doc_field = [[] for __ in TARGET_TEXT_FIELDS]
    ll_aux_doc_att = [[] for __ in TARGET_TEXT_FIELDS]
    l_qid = []
    l_docno_pair = []
    for q, rank in l_q_rank:
        if s_qid is not None:
            if q not in s_qid:
                continue
        logging.info('constructing for q [%s]', q)
        q_boe = h_q_info[q]['query']['boe']
        q_boe = padding(q_boe, q_len)
        q_att = h_q_info[q]['query']['att_mtx']
        for i in xrange(len(rank)):
            docno, doc_score = rank[i]
            doc_info = h_doc_info[docno]
            label = h_qrel.get(q, {}).get(docno, 0)
            if docno not in h_doc_info:
                continue
            for j in xrange(i + 1, len(rank)):
                aux_docno, aux_doc_score = rank[j]
                if aux_docno not in h_doc_info:
                    continue
                aux_doc_info = h_doc_info[aux_docno]
                aux_label = h_qrel.get(q, {}).get(aux_docno, 0)
                if label == aux_label:
                    continue
                l_qid.append(q)
                l_docno_pair.append([docno, aux_docno])
                if label > aux_label:
                    pair_label = 1
                else:
                    pair_label = -1

                l_ltr.append([doc_score])
                l_aux_ltr.append([aux_doc_score])
                l_label.append(pair_label)
                l_q_in.append(q_boe)

                for p in xrange(len(TARGET_TEXT_FIELDS)):
                    field = TARGET_TEXT_FIELDS[p]
                    ll_doc_field[p].append(padding(doc_info[field]['boe'], l_field_len[p]))
                    ll_aux_doc_field[p].append(padding(aux_doc_info[field]['boe'], l_field_len[p]))

                if with_att:
                    l_q_att.append(q_att)
                    for p in xrange(len(TARGET_TEXT_FIELDS)):
                        field = TARGET_TEXT_FIELDS[p]
                        ll_doc_att[p].append(padding_2d(doc_info[field]['att_mtx'], l_field_len[p]))
                        ll_aux_doc_att[p].append(padding_2d(aux_doc_info[field]['att_mtx'], l_field_len[p]))

    if with_att:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field, l_q_att, ll_doc_att)
        x = _add_aux(x, l_aux_ltr, ll_aux_doc_field, ll_aux_doc_att)
    else:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field)
        x = _add_aux(x, l_aux_ltr, ll_aux_doc_field)
    x['qid'] = np.array(l_qid)
    x['docno_pair'] = np.array(l_docno_pair)
    logging.info('pairwise data constructed [%d] q, [%d] pair', len(l_q_in), len(l_label))
    return x, y


def _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field, l_q_att_in=None, ll_doc_att=None):
    x = dict()
    x[q_in_name] = np.array(l_q_in)
    x[ltr_feature_name] = np.array(l_ltr)
    for p in xrange(len(TARGET_TEXT_FIELDS)):
        field = TARGET_TEXT_FIELDS[p]
        x[d_in_name + field] = np.array(ll_doc_field[p])
    y = np.array(l_label)

    if l_q_att_in is not None:
        x[q_att_name] = np.array(l_q_att_in)
        for p in xrange(len(TARGET_TEXT_FIELDS)):
            field = TARGET_TEXT_FIELDS[p]
            x[d_att_name + field] = np.array(ll_doc_att[p])
    return x, y


def _add_aux(x, l_aux_ltr, ll_aux_doc_field, ll_aux_doc_att=None):
    x[aux_pre + ltr_feature_name] = np.array(l_aux_ltr)
    for p in xrange(len(TARGET_TEXT_FIELDS)):
        field = TARGET_TEXT_FIELDS[p]
        x[aux_pre + d_in_name + field] = np.array(ll_aux_doc_field[p])
    if ll_aux_doc_att is not None:
        for p in xrange(len(TARGET_TEXT_FIELDS)):
            field = TARGET_TEXT_FIELDS[p]
            x[aux_pre + d_att_name + field] = np.array(ll_aux_doc_att[p])
    return x


def dump_data(x, y, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(os.path.join(out_dir, 'y'), y)
    for key, arr in x.items():
        np.save(os.path.join(out_dir, key), arr)
    return


def load_data(in_dir, s_target_qid=None):
    x = dict()
    y = None
    for dirname, l_subdir, l_files in os.walk(in_dir):
        for fname in l_files:
            if not fname.endswith('.npy'):
                continue
            key = fname.replace('.npy', '')
            logging.info('get [%s]', key)
            arr = np.load(os.path.join(dirname, fname))
            if key == 'y':
                y = arr
            else:
                x[key] = arr
    if s_target_qid is not None:
        x, y = filter_data(x, y, s_target_qid)
    return x, y


def filter_data(x, y, s_target_qid):
    v_qid = x['qid']
    l_keep_idx = []
    before = y.shape[0]
    for p in xrange(len(v_qid)):
        if v_qid[p] in s_target_qid:
            l_keep_idx.append(p)

    for key in x.keys():
        # if type(x[key]) == list:
        #     x[key] = np.array(x[key])[l_keep_idx].tolist()
        # else:
        x[key] = x[key][l_keep_idx]
    y = y[l_keep_idx]
    after = y.shape[0]
    logging.info('filtered from [%d] to [%d]', before, after)
    return x, y


if __name__ == '__main__':
    from traitlets.config import Configurable
    from traitlets import (
        Unicode,
        Bool,
    )
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log
    )

    class MainPara(Configurable):
        qrel_in = Unicode(help='qrel').tag(config=True)
        q_info_in = Unicode(help='q info tensor').tag(config=True)
        doc_info_in = Unicode(help='doc infor tensor').tag(config=True)
        trec_in = Unicode(help='candidate ranking').tag(config=True)
        out_dir = Unicode(help='out_dir').tag(config=True)
        testing = Bool(False, help='testing').tag(config=True)

    set_basic_log()

    if 2 != len(sys.argv):
        print "convert raw json ts to pairwise and pointwise training data"
        print "1 para, config:"
        MainPara.class_print_help()
        sys.exit(-1)

    para = MainPara(config=load_py_config(sys.argv[1]))
    s_qid = None
    if para.testing:
        s_qid = set(['%s' % i for i in range(1, 11)])  # testing
    pair_x, pair_y = pairwise_reader(para.trec_in, para.qrel_in, para.q_info_in, para.doc_info_in, s_qid)
    logging.info('dumping pairwise x, y')
    dump_data(pair_x, pair_y, os.path.join(para.out_dir, 'pairwise'))

    point_x, point_y = pointwise_reader(para.trec_in, para.qrel_in, para.q_info_in, para.doc_info_in, s_qid)
    logging.info('dumping pointwise x, y')
    dump_data(point_x, point_y, os.path.join(para.out_dir, 'pointwise'))

    if para.testing:
        logging.info('testing load and dump with filter')
        x, y = load_data(os.path.join(para.out_dir, 'pairwise'), {'1'})
        dump_data(x, y, os.path.join(para.out_dir, 'pairwise_test'))
        x, y = load_data(os.path.join(para.out_dir, 'pointwise'), {'1'})
        dump_data(x, y, os.path.join(para.out_dir, 'pointwise_test'))

    logging.info('finished')