"""
count tf of a corpus
input:
    a processed corpus
output:
    tf, sorted via frequency
"""
import sys

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

if 3 != len(sys.argv):
    print "2 para: text in + tf out"
    sys.exit(-1)

h = dict()
for p, line in enumerate(open(sys.argv[1])):
    if not p % 1000:
        print " processed [%d] lines" % p
    for t in line.strip().split():
        if t not in h:
            h[t] = 1
        else:
            h[t] += 1
print "finished reading [%d] lines" % p
print "vocabulary size [%d]" % len(h)
print "sorting..."
l = sorted(h.items(), key=lambda item: -item[1])
out = open(sys.argv[2], 'w')
print "dumping..."
for t, tf in l:
    print >> out, "%s\t%d" % (t, tf)

out.close()
print "tf dumped to [%s]" % sys.argv[2]
