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
    d_att_name
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


def pointwise_reader(trec_in, qrel_in, q_info_in, doc_info_in, s_qid=None, with_att=False):
    logging.info('start read pointwise')
    h_q_info = load_json_info(q_info_in, 'qid')
    h_doc_info = load_json_info(doc_info_in, 'docno')
    l_q_rank = load_trec_ranking_with_score(trec_in)
    h_qrel = load_trec_labels_dict(qrel_in)
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
                ll_doc_field[p].append(doc_info[field]['boe'])

            if with_att:
                l_q_att.append(q_att)
                for p in xrange(len(TARGET_TEXT_FIELDS)):
                    field = TARGET_TEXT_FIELDS[p]
                    ll_doc_att[p].append(doc_info[field]['att_mtx'])
    if with_att:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field, l_q_att, ll_doc_att)
    else:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field)
    x['qid'] = l_qid
    x['docno'] = l_docno
    logging.info('pointwise data constructed [%d] q, [%d] doc', len(l_q_in), len(l_label))
    return x, y


def pairwise_reader(trec_in, qrel_in, q_info_in, doc_info_in, s_qid=None, with_att=False):
    logging.info('start read pairwise')
    h_q_info = load_json_info(q_info_in, 'qid')
    h_doc_info = load_json_info(doc_info_in, 'docno')
    l_q_rank = load_trec_ranking_with_score(trec_in)
    h_qrel = load_trec_labels_dict(qrel_in)
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

    for q, rank in l_q_rank:
        if s_qid is not None:
            if q not in s_qid:
                continue
        logging.info('constructing for q [%s]', q)
        q_boe = h_q_info[q]['query']['boe']
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
                    ll_doc_field[p].append(doc_info[field]['boe'])
                    ll_aux_doc_field.append(aux_doc_info[field]['boe'])

                if with_att:
                    l_q_att.append(q_att)
                    for p in xrange(len(TARGET_TEXT_FIELDS)):
                        field = TARGET_TEXT_FIELDS[p]
                        ll_doc_att[p].append(doc_info[field]['att_mtx'])
                        ll_aux_doc_att[p].append(aux_doc_info[field]['att_mtx'])

    if with_att:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field, l_q_att, ll_doc_att)
        x = _add_aux(x, l_aux_ltr, ll_aux_doc_field, ll_aux_doc_att)
    else:
        x, y = _pack_inputs(l_label, l_q_in, l_ltr, ll_doc_field)
        x = _add_aux(x, l_aux_ltr, ll_aux_doc_field)
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
