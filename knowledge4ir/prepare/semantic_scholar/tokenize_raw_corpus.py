"""
tokenize raw corpus
input:
    s2 corpus in json
    target fields: title and paperAbstract
output:
    id \t title tokens \t abstract tokens
"""

from nltk.tokenize import word_tokenize
import json
import sys

reload(sys)
sys.setdefaultencoding('UTF8')

l_s2_fields=['title', 'paperAbstract']


def tokenize_json_doc(in_name, out_name, l_fields):
    out = open(out_name, 'w')
    for p, line in enumerate(open(in_name)):
        if not p % 1000:
            print 'processed [%d] docs' % p
        h = json.loads(line)
        for field in l_fields:
            h[field] = ' '.join(word_tokenize(' '.join(
                h.get(field, "").split()
                                                       )))
        print >> out, json.dumps(h)
    print "finished"

if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "tokenize s2 corpus to be tagged"
        print "2 para: s2 corpus json + out name"
        sys.exit(-1)
    tokenize_json_doc(sys.argv[1], sys.argv[2], l_s2_fields)

