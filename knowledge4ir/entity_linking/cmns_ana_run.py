"""
run cmns ana
"""

import sys
import json
from knowledge4ir.utils.nlp import raw_clean
import ntpath
from knowledge4ir.entity_linking import CommonEntityLinker
import logging
import os


def _show_conf():
    CommonEntityLinker.class_print_help()
    print 'c.main.text_in=""'
    print 'c.main.out_name=""'
    print 'c.main.in_type="raw|q|doc|corpus"'


def _unit_test(conf):

    linker = CommonEntityLinker(config=conf)
    text_in = conf.main.text_in
    out_name = conf.main.out_name
    out = open(out_name, 'w')

    for line_cnt, line in enumerate(open(text_in)):
        line = line.strip()
        l_ana, text = linker.link(line)
        print >>out, text + '\t#\t' + json.dumps(l_ana)
        if 0 == (line_cnt % 1000):
            logging.info('done %d lines', line_cnt)
    out.close()
    logging.info('done')


def ana_query(conf):

    linker = CommonEntityLinker(config=conf)
    text_in = conf.main.text_in
    out_name = conf.main.out_name
    out = open(out_name, 'w')

    for line_cnt, line in enumerate(open(text_in)):
        line = line.strip()
        qid, query = line.split('\t')[:2]
        h_q_info = json.loads(line.split('#')[-1])
        query = raw_clean(query)
        l_ana, text = linker.link(query, stemming=True)
        h_q_info['ana'] = l_ana
        print >>out, qid + "\t" + query + '\t#\t' + json.dumps(h_q_info)
        if 0 == (line_cnt % 1000):
            logging.info('done %d query', line_cnt)
    out.close()
    logging.info('done')


def ana_text(conf):
    linker = CommonEntityLinker(config=conf)
    text_in = conf.main.text_in
    out_name = conf.main.out_name
    out = open(out_name, 'w')
    col = conf.main.col

    for line_cnt, line in enumerate(open(text_in)):
        line = line.strip()
        text = line.split('\t')[col]
        l_ana, text = linker.link(text, stemming=True)
        print >>out, line + '\t#\t' + json.dumps(l_ana)
        if 0 == (line_cnt % 1000):
            logging.info('done %d text', line_cnt)
    out.close()
    logging.info('done')


def ana_doc(conf, in_name=None):
    linker = CommonEntityLinker(config=conf)
    out_name = conf.main.out_name
    if in_name:
        text_in = in_name
        out_name += '_' + ntpath.basename(in_name)
    else:
        text_in = conf.main.text_in
    out = open(out_name, 'w')

    for line_cnt, line in enumerate(open(text_in)):
        docno, content = line.strip().split('\t')
        content = json.loads(content)
        h_ana = _ana_doc_content(content, linker)
        content.update(h_ana)
        print >>out, docno + '\t' + json.dumps(content)
        if 0 == (line_cnt % 100):
            logging.info('done %d doc', line_cnt)
    out.close()
    logging.info('done')


def _ana_doc_content(content, linker):
    l_text_fields = ['title', 'paperAbstract', 'bodyText']
    kp_field = 'keyPhrases'

    h_field_ana = {}
    h_field_text = {}
    for text_f in l_text_fields:
        if text_f not in content:
            continue
        text = raw_clean(' '.join(content[text_f]))
        l_ana, text = linker.link(text)
        h_field_ana[text_f] = l_ana
        h_field_text[text_f] = text

    l_kp_ana = []
    if kp_field in content:
        l_kp = content[kp_field]
        for kp in l_kp:
            kp = raw_clean(kp)
            l_ana, __ = linker.link(kp)
            l_kp_ana += l_ana
        h_field_ana[kp_field] = l_kp_ana
        h_field_text[kp_field] = l_kp
    h_ana_res = {'ana': h_field_ana, 'clean_text': h_field_text}
    return h_ana_res


def ana_corpus(conf, in_file_name=None):
    linker = CommonEntityLinker(config=conf)
    in_dir = conf.main.in_dir
    out_dir = conf.main.out_dir
    if in_file_name:
        in_name = os.path.join(in_dir, in_file_name)
        out_name = in_name.replace(in_dir, out_dir)
        logging.info('annotating [%s] -> [%s]',
                     in_name,
                     out_name)
        ana_corpus_partition(in_name, out_name, linker)
    else:
        for dirname, subdirnames, filenames in os.walk(in_dir):
            for filename in filenames:
                in_name = os.path.join(dirname, filename)
                out_name = in_name.replace(in_dir, out_dir)
                logging.info('annotating [%s] -> [%s]',
                             in_name,
                             out_name)
                ana_corpus_partition(in_name, out_name, linker)
    logging.info('[%s] annotated to [%s]',
                 in_dir,
                 out_dir,
                 )
    return


def ana_corpus_partition(in_name, out_name, linker):
    out = open(out_name, 'w')
    for cnt, line in enumerate(open(in_name)):
        content = json.loads(line.strip())
        h_field_ana = {}
        h_field_text = {}
        for field, data in content.items():
            text = ""
            if type(data) in {str, unicode}:
                text = data
            if type(data) == list:
                flag = True
                for item in data:
                    if type(item) not in {str, unicode}:
                        flag = False
                        break
                if flag:
                    text = '\t'.join(data)
            text = text.lower()
            l_ana, text = linker.link(text)
            h_field_ana[field] = l_ana
            h_field_text[field] = text
        content.update({'ana': h_field_ana, 'clean_text': h_field_text})
        print >> out, json.dumps(content)
        if 0 == (cnt % 1000):
            logging.info('annotated %d lines', cnt)
    out.close()
    return


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import load_py_config, set_basic_log
    set_basic_log()

    if 2 > len(sys.argv):
        print 'I do simple annotation for given text file'
        print '1+ para. conf + input (if is doc ana)'
        _show_conf()
        sys.exit()
    conf = load_py_config(sys.argv[1])
    if conf.main.in_type == 'raw':
        _unit_test(conf)
    if conf.main.in_type == 'q':
        ana_query(conf)
    if conf.main.in_type == 'doc':
        if len(sys.argv) > 2:
            ana_doc(conf, sys.argv[2])
        else:
            ana_doc(conf)
    if conf.main.in_type == 'corpus':
        if len(sys.argv) > 2:
            ana_corpus(conf, sys.argv[2])
        else:
            ana_corpus(conf)
    if conf.main.in_type == 'text':
        ana_text(conf)



