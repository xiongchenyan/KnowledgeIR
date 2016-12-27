"""
parse web pages in a folder to get their main content
input:
    a dir of htmls
        name: docno.html
do:
    parse with boilerpipe, ArticalExtractor
    only need the body text. Titles are given in the original data.
output:
    one file, each line:
        docno \t title \t bodyText. (\n and \t replaced with " ")
"""

from boilerpipe.extract import Extractor
import os
import sys
import ntpath
reload(sys)
sys.setdefaultencoding('UTF8')




def extract_html(html_text, parser):
    try:
        extractor = Extractor(extractor=parser, html=html_text)
    except (UnicodeDecodeError, TypeError) as e:
        return None, None
    title = extractor.source.getTitle()
    body_text = extractor.getText()
    return title, body_text


def parse_docs(in_name, out_name, parser):
    out = open(out_name, 'w')
    cnt, err_cnt = 0, 0
    for line in open(in_name):
        docno, html_text = line.strip().split('\t')
        html_text = html_text.lower().[html_text.find('<html'):]
        if not html_text:
            continue
        if html_text.strip() == "":
            continue
        if html_text is None:
            continue
        title, body_text = extract_html(html_text, parser)
        if title is None:
            err_cnt += 1
            continue
        print >> out, docno + '\t' + ' '.join(title.strip().split()) + '\t' + ' '.join(body_text.strip().split())
        print '%s parsed' % docno
        cnt += 1

    out.close()
    print "finished %d err, %d correct" % (err_cnt, cnt)


if __name__ == '__main__':
    if 3 > len(sys.argv):
        print "2+ para: html doc in, docno+html one per line + outname + parser "
        print "(default parse: ArticalExtractor, can be: DefaultExtractor, KeepEverythingExtractor, LargestContentExtractor)"
        sys.exit(-1)
    parser = 'ArticleExtractor'
    if len(sys.argv) > 3:
        parser = sys.argv[3]
    parse_docs(sys.argv[1], sys.argv[2], parser)

