"""
average q len (bow and boe)
q len vs relative performance?

input:
    q info
output:
    stats
"""

from knowledge4ir.utils import load_query_info
import json


def avg_len(h_q_info):
    l_bow_len = [len(h['query'].split()) for h in h_q_info]
    l_boe_len = [len(h['tagme']['query']) for h in h_q_info]
    return float(sum(l_bow_len)) / len(l_bow_len), float(sum(l_boe_len)) / len(l_boe_len)


def process(q_info_in, out_name):
    h_q_info = load_query_info(q_info_in)
    bow_len, boe_len = avg_len(h_q_info)
    out = open(out_name, 'w')
    print >> out, 'bow_avg_len: %f\nboe_avg_len: %f' % (bow_len, boe_len)

    out.close()


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "analysis based on query len in bow and boe"
        print "2 para: q info in + out"
        sys.exit(-1)

    process(*sys.argv[1:])
    print "done"


