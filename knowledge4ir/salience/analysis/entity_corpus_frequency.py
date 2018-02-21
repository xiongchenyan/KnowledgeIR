"""
calc entity corpus frequency
input:
    a corpus
output:
    the frequency of entities in the corpus

"""

import json

def count_per_doc(h_doc, h_e_cnt):
    h_ana = h_doc.get('spot', {}).get('bodyText', {})
    if not h_ana:
        h_ana = h_doc.get('spot', {}).get('paperAbstract', {})

    l_e = h_ana.get('entities', [])
    l_feature = h_ana.get('features', [])
    l_tf = [item[0] for item in l_feature]
    for e, tf in zip(l_e, l_tf):
        h_e_cnt[e] = tf + h_e_cnt.get(e, 0)
    return h_e_cnt


def count_corpus_e_freq(in_name, out_name):
    h_e_cnt = dict()
    for p, line in enumerate(open(in_name)):
        if not p % 1000:
            print "counted [%d] lines" % p
        h_doc = json.loads(line)
        h_e_cnt = count_per_doc(h_doc, h_e_cnt)
    json.dump(h_e_cnt, open(out_name, 'w'), indent=1)
    print "done"


if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "count e tf"
        print "2 para: json corpus in + out name"
        sys.exit(-1)

    count_corpus_e_freq(*sys.argv[1:])

