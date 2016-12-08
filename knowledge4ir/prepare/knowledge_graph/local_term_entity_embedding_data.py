"""
generate locally term and entity embedding from PRF docs
input:
    tagme's doc info
output:
    for each doc, two lines:
        body text
        body text with entity replaced
"""

import json
import sys


if 3 != len(sys.argv):
    print "generate locally trained term entity embedding file"
    print "2 para: tagme Doc info + outname"
    sys.exit()

out = open(sys.argv[2], 'w')

for cnt, line in open(sys.argv[1]):
    d_info = json.loads(line.split('\t')[-1])
    text = d_info['bodyText']
    print >> out, text
    e_text = ""
    p = 0
    for ana in d_info['tagme']['bodyText']:
        st, ed = ana[1], ana[2]
        e_text += text[p:st] + ana[0]
        p = ed
    print >> out, e_text

out.close()




