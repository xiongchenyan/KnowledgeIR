"""
hash annotated data
input:
    pickle dict of word->id
    pickle dict of entity->id
    annotated json corpus
do:
    for each fields in each doc/query, hash words, replace unseen words to 0
    for each spot->fields, hash entities, replace unseen ones to 0
"""
import json
import pickle


def hash_per_info(h_info, h_word_id, h_entity_id):
    h_hashed = dict()
    l_field = [field for field in h_info.keys() if field not in {'qid', 'docno', 'spot'}]
    for field in l_field:
        text = h_info[field]
        l_w = text.lower().split()
        l_w_id = [h_word_id.get(w, 0) for w in l_w]
        h_hashed[field] = l_w_id
    for key in ['qid', 'docno']:
        if key in h_info:
            h_hashed[key] = h_info[key]

    h_hashed['spot'] = dict()
    for field, l_ana in h_info['spot'].items():
        l_ana_id = [h_entity_id.get(ana['entities'][0]['id'], 0)
                    for ana in l_ana]
        h_hashed['spot'][field] = l_ana_id
    return h_hashed


def process(json_info_in, word_id_pickle_in, entity_id_pickle_in, out_name):
    h_word_id = pickle.load(open(word_id_pickle_in))
    h_entity_id = pickle.load(open(entity_id_pickle_in))

    out = open(out_name, 'w')
    for p, line in enumerate(open(json_info_in)):
        if not p % 1000:
            print "hashed [%d] lines" % p
        h_info = json.loads(line)
        h_hashed = hash_per_info(h_info, h_word_id, h_entity_id)
        print >> out, json.dumps(h_hashed)

    out.close()
    print "done"


if __name__ == '__main__':
    import sys
    if 4 != len(sys.argv):
        print "hash word and entity in ana info"
        print "3 para: in name + out name + pickle dict pre"
        sys.exit(-1)
    info_in, out_name = sys.argv[1:3]
    word_pickle = sys.argv[3] + '.word.pickle'
    entity_pickle = sys.argv[3] + '.entity.pickle'
    process(info_in, word_pickle, entity_pickle, out_name)

