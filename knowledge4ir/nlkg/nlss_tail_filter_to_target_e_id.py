"""
filter nlss
keep those whose tail is also in given target id set
input:
    nlss
    tail targets
output:
    nlss with tail in targets

"""
import sys
import json

if 4 != len(sys.argv):
    print "filter nlss"
    print "3 para: nlss in + target e id in + out"
    sys.exit(-1)

s_e = set([line.strip().split()[0] for line in open(sys.argv[2])])
s_find = set()
cnt = 0
out = open(sys.argv[3], 'w')
for line in open(sys.argv[1]):
    line = line.strip()
    tail = json.loads(line.split()[-1])
    for tail_e in tail:
        if tail_e in s_e:
            print >> out, line
            cnt += 1
            break

out.close()
print "keep [%d] nlss total" % cnt

