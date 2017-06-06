"""
collect term df, total df, average doc length from a corpus
input:
    directory of the parsed corpus
        (title faked as first 10 terms for now 11/10/2016)
output:
    a df file
    a json of total corpus stats
"""

import os
import pickle


def collect_corpus_stat(corpus_dir, out_pre):
    h_title_df = {}
    h_body_df = {}
    total_tf = 0
    total_df = 0

    for dir_name, sub_dirs, file_names in os.walk(corpus_dir):
        for fname in file_names:
            in_name = os.path.join(dir_name, fname)
            print 'start [%s]' % in_name
            for line_cnt, line in enumerate(open(in_name)):
                if not line_cnt % 1000:
                    print "%d lines" % line_cnt
                l_t = ' '.join(line.strip().split('\t')[2:]).split()
                for t in l_t[:10]:
                    if t not in h_title_df:
                        h_title_df[t] = 1
                    else:
                        h_title_df[t] += 1
                for t in l_t:
                    if t not in h_body_df:
                        h_body_df[t] = 1
                    else:
                        h_body_df[t] += 1
                total_tf += len(l_t)
                total_df += 1
            print '[%s] finished' % in_name

    title_out = open(out_pre + '.title', 'w')
    body_out = open(out_pre + '.bodyText', 'w')
    corpus_out = open(out_pre + '.stat', 'w')
    print "dumping..."
    pickle.dump(h_title_df, title_out)
    pickle.dump(h_body_df, body_out)
    h_corpus_stat = dict()
    h_corpus_stat['title'] = {'total_df': total_df, 'average_len':10}
    h_corpus_stat['bodyText'] = {'total_df': total_df, 'average_len':float(total_tf) / float(total_df)}
    pickle.dump(h_corpus_stat, corpus_out)
    print "finished"
    return


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "I collect corpus stat"
        print "2 para: spot doc dir + output pre"
        sys.exit()
    collect_corpus_stat(*sys.argv[1:])
