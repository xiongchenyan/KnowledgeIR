'''
basic stuff about nlp
'''
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np
import re
import string
from scipy.linalg import norm
regex = re.compile('[%s]' % re.escape(string.punctuation.replace('/', ''))) # see documentation here: http://docs.python.org/2/library/string.html
s_stopwords = set(stopwords.words('english'))


def remove_punctuation(l_words):
    l_new = []
    for word in l_words:
        new_token = regex.sub(u' ', word)
        if not new_token == u'':
            l_new.append(new_token)
    return l_new


def tokenize_and_remove_punctuation(text):
    l_words = word_tokenize(text)
    l_words = remove_punctuation(l_words)
    return l_words


def raw_clean(text):
    return ' '.join(tokenize_and_remove_punctuation(text)).lower()


def rm_stopword(text):
    return ' '.join([t for t in text.split() if t.lower() not in s_stopwords])


def text2lm(text, clean=False):
    if clean:
        text = raw_clean(text)
        text = rm_stopword(text)
        text = text.lower()
    l_term = text.split()
    return term2lm(l_term)


def term2lm(l_term):
    h = {}
    for term in l_term:
        if term in h:
            h[term] += 1
        else:
            h[term] = 1
    return h


def merge_lm(h_lm, h_to_add):
    for term, tf in h_to_add.items():
        if term not in h_lm:
            h_lm[term] = tf
        else:
            h_lm[term] += tf
    return h_lm


def minus_lm(h_lm, h_to_minus):
    for term, tf in h_to_minus.items():
        if term not in h_lm:
            h_lm[term] = -tf
        else:
            h_lm[term] -= tf
    return h_lm


def filter_lm(lm):
    l_term = [item for item in lm.items()
              if (item[1] >= 5)
              & (item[0].lower() not in s_stopwords)
              & (len(item[0]) > 2)
              ]
    return dict(l_term)


def lm_cosine(lm_a, lm_b):
    if (not lm_a) | (not lm_b):
        return 0
    if len(lm_a) > len(lm_b):
        lm_b, lm_a = lm_a, lm_b
    norm_a = norm(lm_a.values())
    norm_b = norm(lm_b.values())
    if (not norm_a) | (not norm_b):
        return 0
    score = 0
    for t, s in lm_a.iteritems():
        score += s * lm_b.get(t, 0)
    return score / norm_a / norm_b


def text_cosine(text_a, text_b):
    lm_a = text2lm(text_a.lower())
    lm_b = text2lm(text_b.lower())
    return lm_cosine(lm_a, lm_b)


def avg_embedding(word2vec, text):
    text = raw_clean(text)
    text = rm_stopword(text)
    l_t = text.split()
    l_v = [word2vec[t] for t in l_t if t in word2vec]
    if not l_v:
        return None
    avg_v = np.mean(np.array(l_v), axis=0)
    return avg_v
