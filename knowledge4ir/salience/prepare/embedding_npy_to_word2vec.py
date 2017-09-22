"""
convert npy data to word2vec format
"""

import json
import numpy as np
import sys
import pickle


def convert_one_line(mtx, k, l_id):
    l = mtx[k].tolist()
    key = l_id[k]
    res = key + ' ' + ' '.join(
        ['%f' for a in l]
    )
    return res


def process(emb_in, id_dict_in, out_name):
    print "loading embedding"
    emb_mtx = np.load(open(emb_in))
    print "loading name id dict"
    h_name_id = pickle.load(open(id_dict_in))
    l_name = [''] * len(h_name_id)
    for name, i in h_name_id.items():
        l_name[i] = name

    out = open(out_name, 'w')
    print "shape %s" % json.dumps(emb_mtx.shape)
    print >> out, '%d %d' % (emb_mtx.shape[0], emb_mtx.shape[1])
    print "dumping.."
    for k in xrange(emb_mtx.shape[0]):
        print >> out, convert_one_line(emb_mtx, k, l_name)
    out.close()
    print "done"


if __name__ == '__main__':
    if 4 != len(sys.argv):
        print "3 para: emb mtx in + id dict in + out name"
        sys.exit(-1)
    process(*sys.argv[1:])
