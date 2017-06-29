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
    for p, line in enumerate(open(in_name)):
        if not p % 1000:
            logging.info('tokenized [%d] lines', p)
        cols = line.split('\t')
        cols = [' '.join(word_tokenize(col)) for col in cols]
        print >> out, '\t'.join(cols)
    out.close()
    logging.info('finished')


if __name__ == '__main__':
    set_basic_log()
    if 3 != len(sys.argv):
        print "2 para: input raw doc (columns splited by tab) + output tokenized doc"
        sys.exit(-1)

    tokenize(*sys.argv[1:])



