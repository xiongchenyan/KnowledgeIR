"""
input:
    this method's trec eva
    base method's trec eva
    q file with qid in the first col
output
    append rel ndcg in the end of q file, with # as seperator

"""

from knowledge4ir.utils import load_gdeval_res


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


def align_rel_ndcg(q_stuff_in, h_q_rel_ndcg, out_name):
    out = open(out_name, 'w')
    for line in open(q_stuff_in):
        line = line.strip()
        q = line.split()[0]
        rel = h_q_rel_ndcg.get(q, 0)
        print >> out, line + '\t#\t%.4f' % rel
    out.close()


def process(eva_res, base_eva_res, q_in, out_name):
    h_q_rel_ndcg = get_rel_ndcg(eva_res, base_eva_res)
    print "ndcg got"

    align_rel_ndcg(q_in, h_q_rel_ndcg, out_name)
    print "done"

if __name__ == '__main__':
    import sys
    if 5 != len(sys.argv):
        print "4 para: eva in + base line eva + q stuff in + out name"
        sys.exit(-1)

    process(*sys.argv[1:])

