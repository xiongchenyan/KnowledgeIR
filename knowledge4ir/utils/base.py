"""
basic functions
"""
import sys
import logging
import logging.handlers
from traitlets.config import PyFileConfigLoader
from knowledge4ir.utils.base_conf import ROOT_PATH
import os
import json
import math
from knowledge4ir.utils.base_conf import TARGET_TEXT_FIELDS
import pickle


def load_qid_query(in_name):
    l_qid_query = [line.split('#')[0].strip().split('\t') for line in open(in_name)]
    return l_qid_query


def load_trec_ranking(in_name):
    """
    Input: trec format ranking results
    :param in_name: the ranking
    :return: ll_qid_ranked_doc [ [qid, [ranked docs]],]
    """
    ll_qid_ranked_doc = []

    this_qid = None

    for line in open(in_name):
        cols = line.strip().split()
        qid = cols[0]
        docno = cols[2]

        if qid != this_qid:
            ll_qid_ranked_doc.append([qid, []])
            this_qid = qid
        ll_qid_ranked_doc[-1][-1].append(docno)
    return ll_qid_ranked_doc


def load_trec_ranking_with_score(in_name):
    ll_qid_ranked_doc = []

    this_qid = None

    for line in open(in_name):
        cols = line.strip().split()
        qid = cols[0]
        docno = cols[2]
        score = float(cols[4])

        if qid != this_qid:
            ll_qid_ranked_doc.append([qid, []])
            this_qid = qid
        ll_qid_ranked_doc[-1][-1].append([docno, score])
    return ll_qid_ranked_doc


def load_trec_ranking_with_info(in_name):
    ll_qid_ranked_doc = []

    this_qid = None

    for line in open(in_name):
        cols = line.strip().split()
        qid = cols[0]
        docno = cols[2]
        score = float(cols[4])
        h_info = json.loads('#'.join(line.split('#')[1:]))

        if qid != this_qid:
            ll_qid_ranked_doc.append([qid, []])
            this_qid = qid
        ll_qid_ranked_doc[-1][-1].append([docno, score, h_info])
    return ll_qid_ranked_doc


def dump_trec_ranking(ll_qid_ranked_doc, out_name):
    out = open(out_name, 'w')
    ll_mid = list(ll_qid_ranked_doc)
    ll_mid.sort(key=lambda item: int(item[0]))
    for l_qid_ranking in ll_mid:
        qid, l_doc = l_qid_ranking
        for p, doc in enumerate(l_doc):
            print >> out, '%s\tQ0\t%s\t%d\t%f # na' % (
                qid,
                doc,
                p + 1,
                -p
            )
    out.close()


def dump_trec_ranking_with_score(ll_qid_ranking, out_name):
    out = open(out_name, 'w')
    ll_mid = list(ll_qid_ranking)
    ll_mid.sort(key=lambda item: int(item[0]))
    for l_qid_ranking in ll_mid:
        qid, l_doc_score = l_qid_ranking
        l_doc_score.sort(key=lambda item: -item[1])
        for p, (doc, score) in enumerate(l_doc_score):
            print >> out, '%s\tQ0\t%s\t%d\t%f # na' % (
                qid,
                doc,
                p + 1,
                score
            )
    out.close()
    logging.info('[%d] query ranking dumped to [%s]', len(ll_qid_ranking), out_name)


def dump_trec_out_from_ranking_score(l_qid, l_docno, l_score, out_name, method_name='na', l_comment=[]):
    l_data = zip(l_qid, zip(l_docno, l_score))
    l_data.sort(key=lambda item: (int(item[0]), -item[1][1]))
    h_doc_comment = dict()
    if l_comment:
        h_doc_comment = dict(zip(l_docno, l_comment))
    out = open(out_name, 'w')
    rank_p = 1
    this_qid = None
    for qid, (docno, score) in l_data:
        if this_qid is None:
            this_qid = qid

        if qid != this_qid:
            rank_p = 1
            this_qid = qid
        res = '%s Q0 %s %d %f # %s' %(
            qid, docno, rank_p, score,
            method_name,
        )
        if l_comment:
            res += ' %s' % h_doc_comment[docno]
        print >> out, res
        rank_p += 1

    out.close()
    logging.debug('ranking result dumped to [%s]', out_name)


def group_scores_to_ranking(l_qid, l_docno, l_score):
    l_data = zip(l_qid, zip(l_docno, l_score))
    l_data.sort(key=lambda item: (int(item[0]), -item[1][1]))

    this_qid = None
    l_q_ranking = []
    for qid, (docno, score) in l_data:
        if qid != this_qid:
            this_qid = qid
            l_q_ranking.append([qid, []])
        l_q_ranking[-1][-1].append((docno, score))
    return l_q_ranking


def load_trec_labels_dict(in_name):
    """
    spot: trec format qrel
    :param in_name:  qrel
    :return: h_qrel = {qid:{doc:score} }
    """
    h_qrel = {}
    l_lines = open(in_name).read().splitlines()
    for line in l_lines:
        cols = line.split()
        qid = cols[0]
        docno = cols[2]
        label = int(cols[3])
        if qid not in h_qrel:
            h_qrel[qid] = {}
        h_qrel[qid][docno] = label

    return h_qrel


def load_trec_labels(in_name):
    h_qrel = load_trec_labels_dict(in_name)
    l_qrel = h_qrel.items()
    l_qrel.sort(key=lambda item: int(item[0]))
    return l_qrel


def dump_trec_labels(l_qrel, out_name):
    out = open(out_name, 'w')
    l_qrel.sort(key=lambda item: int(item[0]))
    for qid, h_doc_score in l_qrel:
        for docno, label in h_doc_score.items():
            print >> out, qid + ' 0 ' + docno + ' ' + str(label)
    out.close()
    logging.debug('[%d] q\'s relevance dumped to [%s]', len(l_qrel), out_name)
    return


def set_basic_log(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def set_log_with_elastic(log_level, out_dir=ROOT_PATH+'/tmp/log/'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    es_logger = logging.getLogger('elasticsearch')
    es_logger.propagate = False
    es_logger.setLevel(logging.INFO)
    es_logger_handler = logging.handlers.RotatingFileHandler(out_dir + '/elastic-base.log',
                                                           maxBytes=0.5*10**7,
                                                           backupCount=10)
    es_logger_handler.setFormatter(formatter)
    es_logger.addHandler(es_logger_handler)

    es_tracer = logging.getLogger('elasticsearch.trace')
    es_tracer.propagate = False
    es_tracer.setLevel(logging.INFO)
    es_tracer_handler=logging.handlers.RotatingFileHandler(out_dir + '/elastic-full.log',
                                                           maxBytes=0.5*10**7,
                                                           backupCount=10)
    # es_tracer_handler.setFormatter(formatter)
    es_tracer.addHandler(es_tracer_handler)

    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(log_level)
    if log_level <= logging.DEBUG:
    # create file handler
        file_handler = logging.handlers.RotatingFileHandler(out_dir + '/full.log',
                                                           maxBytes=10**6,
                                                           backupCount=10)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    lvl = max(log_level, logging.INFO)
    # create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(lvl)
    # create formatter and add it to the handlers

    console_handler.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(console_handler)


def load_py_config(in_name):
    reader = PyFileConfigLoader(in_name)
    reader.load_config()
    logging.info('load from [%s] conf: %s', in_name, reader.config)
    return reader.config


def load_svm_feature(in_name):
    """
    load svm format data
    :param in_name: svm in
    :return: {qid, h_feature, score, and comment}
    """

    l_svm_data = []

    for line in open(in_name):
        line = line.strip()
        cols = line.split('#')
        data = cols[0]
        comment = ""
        if len(cols) > 1:
            comment = '#'.join(cols[1:])
            comment = comment.strip()

        cols = data.split()
        qid = cols[1].replace('qid:', '')
        score = float(cols[0])

        feature_cols = cols[2:]
        l_feature = [item.split(':') for item in feature_cols]
        l_feature = [(int(item[0]), float(item[1])) for item in l_feature]
        h_feature = dict(l_feature)
        l_svm_data.append({
            'qid': qid,
            'score': score,
            'feature': h_feature,
            'comment': comment
        })
    logging.info('load [%d] svm data line from [%s]', len(l_svm_data), in_name)
    return l_svm_data


def add_svm_feature(h_feature_a, h_feature_b):
    h_res = dict(h_feature_a)
    for key, score in h_feature_b.items():
        h_res[key] = h_res.get(key, 0) + score
    return h_res


def dump_svm_feature(l_svm_data, out_name):
    """
    svm_data:
            'qid': qid,
            'score': score,
            'feature': h_feature,
            'comment': comment
    :param l_svm_data:
    :param out_name:
    :return:
    """
    out = open(out_name, 'w')
    l_svm_data.sort(key=lambda item: int(item['qid'])) # sort
    for svm_data in l_svm_data:
        print >>out, _dumps_svm_line(svm_data)
    out.close()
    logging.info('dump [%d] svm line to [%s]', len(l_svm_data), out_name)
    return


def _dumps_svm_line(svm_data):
    res = '%d qid:%s' % (int(svm_data['score']), svm_data['qid'])
    l_feature = svm_data['feature'].items()
    l_feature.sort(key=lambda item: int(item[0]))
    l_feature_str = ['%d:%.6f' % (item[0], item[1]) for item in l_feature]
    res += ' ' + ' '.join(l_feature_str)
    res += ' # ' + svm_data['comment']
    return res


def dump_svm_from_raw(out_name, l_qid, l_docno, l_score, l_h_feature):
    out = open(out_name, 'w')
    l_h_hash_feature, h_feature_name = feature_hash(l_h_feature)
    for p in range(len(l_qid)):
        svm_data = dict()
        svm_data['score'] = l_score[p]
        svm_data['qid'] = l_qid[p]
        svm_data['feature'] = l_h_hash_feature[p]
        svm_data['comment'] = l_docno[p]
        try:
            print >> out, _dumps_svm_line(svm_data)
        except UnicodeEncodeError:
            continue
    out.close()
    return h_feature_name


def feature_hash(l_h_feature):
    """
    hash feature id
    :param l_h_feature: list of feature
    :return: hashed feature, together with feature name
    """
    l_h_hashed_feature = []
    h_name = {}
    l_name = []
    for h_feature in l_h_feature:
        l_name.extend(h_feature.keys())
    l_name = list(set(l_name))
    l_name.sort()
    if 'base' in l_name:
        l_name = ['base'] + [name for name in l_name if name != 'base']
    for name in l_name:
        if name not in h_name:
            h_name[name] = len(h_name) + 1
    for h_feature in l_h_feature:
        h_new_feature = {}
        for name, score in h_feature.items():
            name = h_name[name]
            h_new_feature[name] = score
        l_h_hashed_feature.append(h_new_feature)
    return l_h_hashed_feature, h_name


def load_gdeval_res(in_name, with_mean=True):
    return seg_gdeval_out(open(in_name).read(), with_mean)


def seg_gdeval_out(eva_str, with_mean=True):
    l_qid_eva = []
    mean_ndcg = 0
    mean_err = 0
    for line_cnt, line in enumerate(eva_str.splitlines()):
        if line_cnt == 0:
            continue
        qid, ndcg, err = line.split(',')[-3:]
        ndcg = float(ndcg)
        err = float(err)
        if qid == 'amean':
            mean_ndcg = ndcg
            mean_err = err
        else:
            l_qid_eva.append([qid, (ndcg, err)])
    # logging.info('get eval res %s, mean %f,%f', json.dumps(l_qid_eva), mean_ndcg, mean_err)
    l_qid_eva.sort(key=lambda item: int(item[0]))
    if with_mean:
        return l_qid_eva, mean_ndcg, mean_err
    else:
        return l_qid_eva


def get_rel_ndcg(eva_res, base_eva_res):
    l_q_eva, __, __ = load_gdeval_res(eva_res)
    l_base_q_eva, __, __ = load_gdeval_res(base_eva_res)
    h_q_rel_ndcg = dict()
    h_base_q_eva = dict(l_base_q_eva)
    for q, (ndcg, __) in l_q_eva:
        base_ndcg = h_base_q_eva.get(q, [0, 0])[0]
        rel = ndcg - base_ndcg
        h_q_rel_ndcg[q] = rel

    return h_q_rel_ndcg


def rm3(ranking, l_doc_h_tf, l_doc_h_df=None, total_df=None, h_total_df=None, normalize=True):
    """
    rm3 model
    if h_doc_df and total_df is None, will only use tf part
        \sum_d tf*doc score
    else:
        \sum_d tf*log(idf) * doc score
    :param ranking: [(doc,ranking score),..]
    :param l_doc_h_tf: tf dict for each doc
    :param l_doc_h_df: df dict for each doc
    :param total_df: total df of the corpus
    :param h_total_df: total df dict
    :param normalize: whether normalize tf
    :return: expansion term with score in a list, [[term, exp score],...]
    """
    h_term_score = {}
    assert len(ranking) == len(l_doc_h_tf)
    if l_doc_h_df:
        assert len(ranking) == len(l_doc_h_df)
    for p in xrange(len(ranking)):
        score = ranking[p][1]
        h_tf = l_doc_h_tf[p]
        h_df = {}
        if l_doc_h_df:
            h_df = l_doc_h_df[p]
        if h_total_df:
            h_df = h_total_df
        tf_z = float(sum([item[1] for item in h_tf.items()]))
        for term, tf in h_tf.items():
            if normalize:
                exp_score = tf / tf_z * score
            else:
                exp_score = tf * score
            if h_df:
                idf = 0.5
                if term in h_df:
                    idf = float(total_df) / h_df[term]
                exp_score *= math.log(idf)
            if term not in h_term_score:
                h_term_score[term] = exp_score
            else:
                h_term_score[term] += exp_score
    l_exp_term = sorted(h_term_score.items(), key=lambda item: -item[1])
    return l_exp_term


def bin_similarity(l_sim, l_bins, bin_func='log'):
    l_bin_nb = [0] * len(l_bins)
    for p in xrange(len(l_sim)):
        for bin_p in xrange(len(l_bins)):
            if l_sim[p] >= l_bins[bin_p]:
                l_bin_nb[bin_p] += 1
                break
    if bin_func == 'log':
        l_bin_nb = [math.log(max(score, 1e-10)) for score in l_bin_nb]
    elif bin_func == 'norm_tf':
        z = float(sum(l_bin_nb))
        if z:
            l_bin_nb = [score / z for score in l_bin_nb]
    l_names = ['bin_%d' % i for i in xrange(len(l_bins))]
    return zip(l_names, l_bin_nb)


def form_bins(nb_bin, bin_range=1):
    l_bins = [1]
    if nb_bin == 1:
        return l_bins
    bin_size = bin_range / float(nb_bin - 1)
    for i in xrange(nb_bin - 1):
        bound = l_bins[i] - bin_size
        l_bins.append(bound)
    logging.debug('using bin [%s]', json.dumps(l_bins))
    return l_bins


def load_q_ana(in_name):
    h_q_e = {}
    for line in open(in_name):
        q_data, info = line.strip().split('#')
        qid = q_data.split()[0]
        h = json.loads(info)
        l_e = [ana[0] for ana in h['ana']]
        h_q_e[qid] = l_e
    logging.info('loaded ana for [%d] query', len(h_q_e))
    return h_q_e


def load_doc_ana(in_name, s_target=None):
    h_d_e = {}
    logging.info('start loading doc ana...')
    for cnt, line in enumerate(open(in_name)):
        if 0 == (cnt % 1000):
            logging.info('loaded [%d] docs', cnt)
        docno, info = line.strip().split('\t')
        if s_target is not None:
            if docno not in s_target:
                continue
        content = json.loads(info)
        h_field_e = {}
        for field, l_ana in content['ana'].items():
            h_field_e[field] = [(ana[0], ana[-1]) for ana in l_ana]
        h_d_e[docno] = h_field_e

    logging.info('loaded ana for [%d] documents', len(h_d_e))
    return h_d_e


def load_query_info(in_name):
    """
    read what is output in batch_get_query_info
    :param in_name:
    :return:
    """
    l_lines = open(in_name).read().splitlines()
    l_vcol = [line.split('\t') for line in l_lines]
    l_qid = [vcol[0] for vcol in l_vcol]
    l_h_q_info = [json.loads(vcol[-1]) for vcol in l_vcol]

    return dict(zip(l_qid, l_h_q_info))


def load_doc_info(in_name):
    """
    :param in_name:
    :return:
    """
    logging.info('start loading doc info %s', in_name)
    l_lines = open(in_name).read().splitlines()
    logging.info('total [%d] docs', len(l_lines))
    l_vcol = [line.split('\t') for line in l_lines]
    l_docno = [vcol[0] for vcol in l_vcol]
    l_h_doc_info = [json.loads(vcol[-1]) for vcol in l_vcol]
    h_doc_info = dict(zip(l_docno, l_h_doc_info))
    logging.info('loaded [%d] doc info', len(h_doc_info))
    return h_doc_info


def load_json_info(in_name, key_field='docno', unpack=True):
    """
    load json format info file
    :param in_name: input, each line is a json (dict)
    :param key_field: the field to index
    :param unpack: whether to unpack json lines
    :return: h_info key->dict
    """
    logging.info('loading [%s] with key [%s]', in_name, key_field)
    if unpack:
        l_h = [json.loads(line) for line in open(in_name)]
        l_key = [h[key_field] for h in l_h]
        h_info = dict(zip(l_key, l_h))
    else:
        logging.info('delay unpack json string at run time, return key->line mapping')
        h_info = {}
        for line in open(in_name):
            h_info[json.loads(line)[key_field]] = line.strip()
    logging.info('loaded [%d] info', len(h_info))
    return h_info


def load_corpus_stat(in_pre, l_field=TARGET_TEXT_FIELDS):
    l_field_h_df = []
    logging.info('start loading corpus stat from [%s], target field %s',
                 in_pre, json.dumps(l_field))
    for field in l_field:
        if os.path.exists(in_pre + '.' + field):
            logging.info('loading df from [%s]', in_pre + '.' + field)
            h_df = pickle.load(open(in_pre + '.' + field))
            l_field_h_df.append((field, h_df))

    h_corpus_stat = pickle.load(open(in_pre + '.stat'))
    logging.info('corpus stats loaded')
    return l_field_h_df, h_corpus_stat


def mean_pool_feature(l_h_feature, add_suffix=True):
    h_res = dict()
    z = float(len(l_h_feature))
    for h_feature in l_h_feature:
        for key, v in h_feature.items():
            h_res[key] = v / z + h_res.get(key, 0)
    if add_suffix:
        return add_feature_suffix(h_res, '_Mean')
    else:
        return h_res


def max_pool_feature(l_h_feature, add_suffix=True):
    h_res = dict()
    z = float(len(l_h_feature))
    for h_feature in l_h_feature:
        for key, v in h_feature.items():
            h_res[key] = max(v, h_res.get(key, -10000000))
    # h_res = dict([(item[0] + '_Max', item[1]) for item in h_res.items()])
    if add_suffix:
        return add_feature_suffix(h_res, '_Max')
    else:
        return h_res


def sum_pool_feature(l_h_feature, add_suffix=True):
    h_res = dict()
    for h_feature in l_h_feature:
        for key, v in h_feature.items():
            h_res[key] = v + h_res.get(key, 0)
    if add_suffix:
        return add_feature_suffix(h_res, '_Sum')
    else:
        return h_res


def log_sum_feature(l_h_feature):
    h_res = dict()
    for h_feature in l_h_feature:
        for key, v in h_feature.items():
            h_res[key + "_LogSum"] = math.log(max(v, math.exp(-20))) + h_res.get(key, 0)
    return h_res


def add_feature_prefix(h_feature, prefix):
    h_new = dict(
        [(prefix + key, value) for key, value in h_feature.items()]
    )
    return h_new


def add_feature_suffix(h_feature, suffix):
    h_new = dict(
        [(key + suffix, value) for key, value in h_feature.items()]
    )
    return h_new
