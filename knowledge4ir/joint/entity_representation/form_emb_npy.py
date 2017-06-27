"""
convert word2vec embedding to npy
input:
    word2vec format entity embedding
    e_id -> idx json (0 is mask)
output:
    npy format embedding mtx
"""

import json
import numpy as np
import sys

if 4 != len(sys.argv):
    print "3 para: word2vec + entity id -> idx json + output emb name"
    sys.exit(-1)

h_e_idx = json.load(open(sys.argv[2]))
print "%d entity" % len(h_e_idx)
filled_cnt = 0
emb_mtx = np.array([])
for p, line in enumerate(open(sys.argv[1])):
    if not p:
        __, dim = [int(col) for col in line.strip().split()]
        print "%d dim" % dim
        emb_mtx = np.zeros((len(h_e_idx), dim))
        continue
    if not p % 1000:
        print "processed [%d] lines [%d] filled" % (p, filled_cnt)

    cols = line.split()
    e_id = cols[0]
    if e_id not in h_e_idx:
        continue
    emb = [float(col) for col in cols[1:]]
    idx = h_e_idx[e_id]
    emb_mtx[idx] = np.array(emb)

print "emb mtx shape"
print emb_mtx.shape
print "%d/%d [%.2f%%] filled" % (filled_cnt, len(h_e_idx) - 1,
                                 100.0 * filled_cnt / float(len(h_e_idx) - 1)
                                 )
print "dumping..."
np.save(sys.argv[3], emb_mtx)
print "finished"
