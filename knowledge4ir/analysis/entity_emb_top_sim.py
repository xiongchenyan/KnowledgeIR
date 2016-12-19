"""
get top sim entities and see if they are reasonable
input:
    target entities, one per line
    entity texts in json format (will only use name)
    entity embedding in word2vec format
output:
    top sim entities for each target
    format:
        target id \t target name \t sim id \t sim name \t cosine score

"""

from gensim.models import Word2Vec
import json
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def load_entity_names(text_in):
    h_e_name = {}
    for line in open(text_in):
        h = json.loads(line)
        h_e_name[h['id']] = h['name']
    print "entity names loaded"
    return h_e_name


def get_top_sim_per_e(e, emb, h_e_name):
    if e not in emb:
        return []
    l_e = emb.most_similar([e], topn=10)
    lines = []

    e_name = h_e_name.get(e, 'NA')
    for sim_e, score in l_e:
        name = h_e_name.get(sim_e, 'NA')
        res = '%s\t%s\t%s\t%s\t%f' % (e, e_name, sim_e, name, score)
        lines.append(res)
    print '[%s][%s] top 10 sim get' % (e, e_name)
    return lines


def get_top_sim(target_in, emb_in, text_in, out_name):
    h_e_name = load_entity_names(text_in)
    print 'loading embedding...'
    emb = Word2Vec.load_word2vec_format(emb_in)
    print 'getting top sim'

    out = open(out_name, 'w')
    for e in open(target_in):
        e = e.strip()
        res_lines = get_top_sim_per_e(e, emb, h_e_name)
        if res_lines
            print >> out, '\n'.join(res_lines)
    out.close()
    print "done"

if __name__ == '__main__':
    if 5 != len(sys.argv):
        print "get top sim entities"
        print "4 para: target entities + embedding + entity text + out"
        sys.exit(-1)
    get_top_sim(*sys.argv[1:])



