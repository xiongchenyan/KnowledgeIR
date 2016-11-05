"""
filter embedding to given vocabulary
input:
    word2vec
    vocabulary (first col is the vocabulary)
output:
    word2vec in filtered format
"""

import sys


def filter_embedding(word2vec_in, voc_in, out_name):
    s_voc = set([line.split('\t')[0] for line in open(voc_in)])
    print "%d vocabulary" % len(s_voc)

    total_cnt, dim = 0, 0
    in_cnt = 0
    print "Start counting new vocabulary size..."
    for line_cnt, line in enumerate(open(word2vec_in)):
        if not line_cnt:
            total_cnt, dim = line.split()
            total_cnt = int(total_cnt)
            dim = int(dim)
            continue
        if 0 == (line_cnt % 1000):
            print "checked %d line [%d] kept" % (line_cnt, in_cnt)
        t = line.split()[0]
        print "this line term is [%s]" % t
        if t in s_voc:
            in_cnt += 1

    print "[%d/%d] in embedding" % (in_cnt, len(s_voc))
    print "start filtering"
    out = open(out_name, 'w')

    print >>out, "%d %d" % (in_cnt, dim)
    for line_cnt, line in enumerate(open(word2vec_in)):
        if not line_cnt:
            continue
        if 0 == (line_cnt % 1000):
            print "filtered %d line" % line_cnt

        t = line.split()[0]
        if t in s_voc:
            print >> out, line.strip()

    out.close()
    print 'finished'


if __name__ == '__main__':
    if 4 != len(sys.argv):
        print "3 para: word2vec in + vocabulary + out"
        print "will filter embedding to given vocabulary"
        sys.exit()

    filter_embedding(*sys.argv[1:])
