import json
import numpy as np
import pickle
import gzip


def get_event_vocab(dataset):
    vocab = set()
    open_func = gzip.open if dataset.endswith("gz") else open
    with open_func(dataset) as fin:
        count = 0
        for line in fin:
            data = json.loads(line)
            for spot in data['event']['bodyText']:
                for feature in spot['feature']['sparseFeatureArray']:
                    if feature.startswith("LexicalHead"):
                        vocab.add(feature.split("_", 1)[1])
            count += 1
            if not count % 10000:
                print "Read [%d] training data." % count
    return vocab


def process(train_dataset, in_name, out_pre):
    vocab = get_event_vocab(train_dataset)
    print "Event vocabulary size [%d]" % (len(vocab))

    l_word_emb = []
    l_word = []
    v_size, d = 0, 300

    for p, line in enumerate(open(in_name)):
        if not p:
            v_size, d = line.strip().split()
            d = int(d)
            l_word.append('UNK')
            l_word_emb.append(np.random.rand(d))
            continue
        if not p % 10000:
            print "Read [%d] lines of embeddings" % p

        cols = line.strip().split()
        v = cols[0]
        emb = [float(col) for col in cols[1:]]
        assert len(emb) == d
        if v in vocab:
            l_word.append(v)
            l_word_emb.append(emb)

    print "[%d] events extracted from embedding out of [%d]" % (
        len(l_word), len(vocab))
    print "dumping event hash..."
    h_w = dict(zip(l_word, range(len(l_word))))
    pickle.dump(h_w, open(out_pre + '.event.pickle', 'w'))
    print "dumping event emb..."
    mtx_emb = np.array(l_word_emb)
    np.save(open(out_pre + '_event_emb.npy', 'w'), mtx_emb)

    print "finished"


if __name__ == '__main__':
    import sys

    if 4 != len(sys.argv):
        print "make embedding event vocabulary from training and then hash"
        print "3 para: training data + word2vec in + out pre"
        sys.exit(-1)

    process(*sys.argv[1:])
