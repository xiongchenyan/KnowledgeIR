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
        l_text = []
        for field in l_fields:
            text = ' '.join(word_tokenize(' '.join(
                h.get(field, "").split()
                                                       )))
            l_text.append(text)
        docno = h.get('id', "")
        if not docno:
            docno = h.get('docno', "")
        if docno:
            print >> out, docno + '\t' + '\t'.join(l_text)
    out.close()
    print "finished"


if __name__ == '__main__':
    if 3 != len(sys.argv):
        print "tokenize s2 corpus to be tagged"
        print "2 para: s2 corpus json + out name"
        sys.exit(-1)
    tokenize_json_doc(sys.argv[1], sys.argv[2], l_s2_fields)

