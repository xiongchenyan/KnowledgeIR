"""
get robust04 documents
input: trec ranking file + TREC CD directory
do:
    for each .dat in the directory:
        for each record in the xml tree:
            get docno, headline, and text
            if docno in the trec ranking file:
                print headline and text to each text
out:
    one file for each textual field of candidate documents
        docno \t headline(title)|body

"""

import xml.etree.ElementTree as ET
import os
from nltk.tokenize import word_tokenize
from knowledge4ir.utils import load_trec_ranking
import sys

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


TITLE_FIELD = "HEADLINE"
ID_FIELD = "DOCNO"
BODY_FIELD = "TEXT"
SUFFIX = '.dat'


def parse_one_trec_xml_file(in_name, s_target_docno):
    print "start processing [%s]" % in_name
    l = open(in_name).read()
    l = l.replace("&", "&#038;")
    l = "<root>" + l + "</root>"
    xml_tree = ET.fromstring(l)
    cnt = 0
    l_doc_title = []
    l_doc_body = []
    for doc in xml_tree:
        cnt += 1
        docno = doc.find(ID_FIELD).text.strip()
        if docno not in s_target_docno:
            continue
        title = doc.find(TITLE_FIELD).text.strip()
        body_text = doc.find(BODY_FIELD).text.strip()
        title = ' '.join(word_tokenize(title))
        body_text = ' '.join(word_tokenize(body_text))
        l_doc_title.append((docno, title))
        l_doc_body.append((docno, body_text))
    print "[%s] file, [%d/%d] are target docs" % (in_name, len(l_doc_title), cnt)
    return l_doc_title, l_doc_body


def process_directory(in_dir, out_pre, s_target_docno):
    title_out = open(out_pre + '.title', 'w')
    body_out = open(out_pre + '.bodyText', 'w')
    find_cnt = 0
    for dir_name, sub_dirs, file_names in os.walk(in_dir):
        for f_name in file_names:
            if f_name.endswith(SUFFIX):
                in_name = os.path.join(dir_name, f_name)
                l_doc_title, l_doc_body = parse_one_trec_xml_file(in_name, s_target_docno)
                find_cnt += len(l_doc_title)
                for docno, title in l_doc_title:
                    print >> title_out, docno + '\t' + title
                for docno, body in l_doc_body:
                    print >> body_out, docno + '\t' + body
    title_out.close()
    body_out.close()
    print "[%s] directory finished [%d/%d] target docs found" % (in_dir, find_cnt, len(s_target_docno))
    return


def get_target_doc(in_dir, out_pre, trec_rank_in):
    l_rank = load_trec_ranking(trec_rank_in)
    s_target_docno = set(sum([item[1] for item in l_rank], []))
    print "total [%d] target docno" % len(s_target_docno)
    process_directory(in_dir, out_pre, s_target_docno)


if __name__ == '__main__':
    if 4 != len(sys.argv):
        print "3 para: trec cds directory, output prefix, trec rank file"
        sys.exit(-1)

    get_target_doc(*sys.argv[1:])







