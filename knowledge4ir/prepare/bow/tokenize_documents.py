"""
tokenize doc
(keep punctuations)
"""

from nltk.tokenize import word_tokenize
import sys
import logging
from knowledge4ir.utils import (
    set_basic_log
)

reload(sys)
sys.setdefaultencoding('UTF8')


def tokenize(in_name, out_name):
    out = open(out_name, 'w')
    cnt = 0
    err_cnt = 0
    for p, line in enumerate(open(in_name)):
        if not p % 1000:
            logging.info('tokenized [%d] lines', p)
        cols = line.split('\t')
        cnt += 1
        try:
            cols = [cols[0]] + [' '.join(word_tokenize(col)) for col in cols[1:]]
            print >> out, '\t'.join(cols)
        except UnicodeDecodeError:
            p += 1
            print "DECODE ERROR:"
            print line

    out.close()
    logging.info('finished, [%d/%d] error', err_cnt, cnt)


if __name__ == '__main__':
    set_basic_log()
    if 3 != len(sys.argv):
        print "2 para: input raw doc (columns splited by tab) + output tokenized doc"
        sys.exit(-1)

    tokenize(*sys.argv[1:])



