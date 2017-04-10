"""
in: words or phrases, one per line
in: word2vec model, Google format
out: words\t [similar words]
"""
from gensim.models import Word2Vec

import sys
import json
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


if len(sys.argv) != 4:
    print "I get similar words for given words and model"
    print '3 para: target words, one per line + google word2vec trained model + out'
    sys.exit()

print 'start loading the model, will take a while...'
model = Word2Vec.load_word2vec_format(sys.argv[2])

print 'model loaded'
l_words = open(sys.argv[1]).read().splitlines()

out = open(sys.argv[3], 'w')

for words in l_words:
    l = words.split()
    try:
        res = model.most_similar(l, topn=100)
        print >> out, words + '\t' + json.dumps(res)
        print '[%s] get' % words
    except KeyError:
        print >> out, words + '\t\"OOV\"'
        continue

print 'all get'
out.close()

