"""
split doc info via candidate file partition
input:
    folder of trec splits
    doc info in
    out folder
output:
    out folder/[doc_info].[trec splits name]
"""
import os
import sys
import json
import logging
import ntpath
from knowledge4ir.utils import (
    load_trec_ranking_with_score,
    load_json_info,
    set_basic_log,
)


def mul_load_candidate_doc(in_dir):
    l_name, l_s_id = [], []
    logging.info('load doc partitions')
    for dir_name, sub_dir, f_names in os.walk(in_dir):
        for f_name in f_names:
            l_name.append(f_name)
            l_q_rank = load_trec_ranking_with_score(os.path.join(dir_name, f_name))
            s_doc = set(sum([ [doc for doc, score in rank] for q, rank in l_q_rank ], []))
            s_qid = set([q for q, __ in l_q_rank])
            s_id = s_doc.union(s_qid)
            l_s_id.append(s_doc.union(s_qid))
            logging.info('[%s][%d] doc', f_name, len(s_id))
    return l_name, l_s_id


def load_both_info(json_info_in):
    l_id = ['qid', 'docno']
    h_info = {}
    for line in open(json_info_in):
        h = json.loads(line)
        d_id = ''.join([h[key] for key in l_id])
        h_info[d_id] = h
    return h_info


def partition_json_info(json_info_in, l_name, l_s_id, out_dir):
    logging.info('start partion json info')
    h_info = load_both_info(json_info_in)
    for name, s_id in zip(l_name, l_s_id):
        out_name = os.path.join(
            out_dir, ntpath.basename(json_info_in)) + '.' + name.split('.')[-1]
        out = open(out_name, 'w')
        logging.info('[%s] has [%d] target doc', name, len(s_id))
        cnt = 0
        for data_id in s_id:
            if data_id in h_info:
                cnt += 1
                print >> out, json.dumps(h_info[data_id])
        logging.info('[%s] finished [%d/%d] find', out_name, cnt, len(s_id))
        out.close()
    logging.info('finished')

if __name__ == '__main__':
    set_basic_log()
    if 4 != len(sys.argv):
        print "split json info for given data"
        print "3 para: json info in, splited trec dir, out dir"
        sys.exit(-1)

    json_info_in, trec_in_dir, out_dir = sys.argv[1:4]
    l_name, l_s_doc = mul_load_candidate_doc(trec_in_dir)
    partition_json_info(json_info_in, l_name, l_s_doc, out_dir)
