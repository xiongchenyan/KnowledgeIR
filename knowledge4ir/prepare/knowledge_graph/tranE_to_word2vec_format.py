"""
transfer transE format entity embedding to Google word2vec format
input:
    entity2id
    entity2vec
output:
    tranE embedding with work2vec format
"""

import sys

if 4 != len(sys.argv):
    print "3 para: entity2id + entity2vec + out name"
    sys.exit(-1)

lines = open(sys.argv[1]).read().splitlines()
total_cnt = int(lines[0])
l_e = [line.split()[0].replace('.', '/') for line in lines[1:]]
assert len(l_e) == total_cnt

out = open(sys.argv[3], 'w')
print >> out, '%d 50' % total_cnt

for line_cnt, line in enumerate(open(sys.argv[2])):
    line = ' '.join(line.strip().split())
    print >> out, l_e[line_cnt] + ' ' + line
    if not line_cnt % 10000:
        print "processed [%d] lines" % line_cnt

out.close()
print "finished"

