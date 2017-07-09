"""
split corpus stat
input:
    corpus stat in
    vocab in
output:
    corpus stat in.0x, df restricted to vocab in

"""

import json
from knowledge4ir.utils.resource import CorpusStat
import sys
import ntpath

if 4 != len(sys.argv):
    print "3 para: corpus start in + vocab in + out pre"
    sys.exit(-1)

corpus_stat = CorpusStat()
corpus_stat.load(sys.argv[1])

vocab_in = sys.argv[2]
out_pre = sys.argv[3]

suf = vocab_in.replace('.vocab', '').split('.')[-1]
print "suffix: %s" % suf


vocab = set([line.strip().split('\t')[0] for line in open(sys.argv[2])])
print "[%d] target vocab" % len(vocab)

h_new_field_df = dict()

for field, h_df in corpus_stat.h_field_df.items():
    h_new_df = dict([(v, df) for v, df in h_df.items() if v in vocab])
    h_new_field_df[field] = h_new_df
    print '[%s] [%d]->[%d]' % (field, len(h_df), len(h_new_field_df))

corpus_stat.h_field_df = h_new_field_df
corpus_stat.dump(sys.argv[3] + '.' + suf)

