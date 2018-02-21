import sys
import json

in_file = sys.argv[1]
out_file = sys.argv[2]

h_idf = {}
num_words = 0
num_events = 0
num_doc = 0
num_salience = 0

with open(in_file) as f:
    for line in f:
        info = json.loads(line)
        num_doc += 1
        words = info['bodyText']
        num_words += len(words)
        event_heads = info['event']['bodyText']['sparse_features'].get(
            'LexicalHead', [])
        labels = info['event']['bodyText']["salience"]
        num_salience += sum(labels)
        num_events += len(event_heads)
        for e in set(event_heads):
            try:
                h_idf[e] += 1
            except KeyError:
                h_idf[e] = 1

print("Number words per doc: %.4f, number events per doc"
      ": %.4f, number doc: %d, number salience: %.2f" % (
          1.0 * num_words / num_doc, 1.0 * num_events / num_doc, num_doc,
          1.0 * num_salience / num_doc
      ))

import operator

sorted_idf = sorted(h_idf.items(), key=operator.itemgetter(1), reverse=True)

with open(out_file, 'w') as out:
    for k, v in sorted_idf:
        out.write("%s\t%d\n" % (k, v))
