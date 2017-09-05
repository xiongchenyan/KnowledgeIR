"""
for term hash and embedding numpy mtx
input:
    word2vec format embedding
output:
    word -> id's pickle dict
    entity -> id's pickle dict
    word embedding mtx, np.save() format, each row corresponds to word's id (start from 0's unk embedding)
    entity embedding mtx, one row for each entity, start from 0's unk embedding
"""

import pickle
import numpy as np
import json
import logging


def process(in_name, out_pre):

    l_word_emb = []
    l_e_emb = []
    l_word = []
    l_e = []
    v_size, d = 0, 300

    for p, line in enumerate(open(in_name)):

        if not p:
            v_size, d = line.strip().split()
            d = int(d)
            l_word.append('UNK')
            l_e.append('/m/UNK')
            l_word_emb.append(np.random.rand(d))
            l_e_emb.append(np.random.rand(d))
            continue
        if not p % 10000:
            print "read [%d] lines" % p

        cols = line.strip().split()
        v = cols[0]
        emb = [float(col) for col in cols[1:]]
        assert len(emb) == d
        if v.startswith('/m/'):
            l_e.append(v)
            l_e_emb.append(emb)
        else:
            l_word.append(v)
            l_word_emb.append(emb)

    print "[%d] word [%d] e" % (len(l_word), len(l_e))
    print "dumping word hash..."
    h_w = dict(zip(l_word, range(len(l_word))))
    pickle.dump(h_w, open(out_pre + '.word.pickle', 'w'))
    print "dumping entity hash..."
    h_e = dict(zip(l_e, range(len(l_e))))
    pickle.dump(h_e, open(out_pre + '.entity.pickle', 'w'))
    print "dumping word emb..."
    mtx_emb = np.array(l_word_emb)
    np.save(open(out_pre + 'word_emb.npy', 'w'), mtx_emb)
    print "dumping entity emb..."
    mtx_emb = np.array(l_e_emb)
    np.save(open(out_pre + 'entity_emb.npy', 'w'), mtx_emb)

    print "finished"


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "make embedding npy mtx and vocabulary hash"
        print "2 para: word2vec in + out pre"
        sys.exit(-1)

    process(*sys.argv[1:])





