"""
prepare corpus stat from input doc info
input:
    doc info json
do:
    count title len, bodyText len, and df
output:
    pickle dumped corpus_stat
"""

import json
import sys
from knowledge4ir.utils import TARGET_TEXT_FIELDS
import pickle

if 3 != len(sys.argv):
    print "2 para: doc info + corpus stat out"
    sys.exit(-1)

h_field_cnt = dict()
h_field_len = dict()
h_field_df = dict()
for field in TARGET_TEXT_FIELDS:
    h_field_cnt[field] = 0
    h_field_len[field] = 0
    h_field_df[field] = dict()

for p, line in enumerate(open(sys.argv[1])):
    if not p % 1000:
        print "processed [%d] doc" % p

    h_info = json.loads(line)
    for field in TARGET_TEXT_FIELDS:
        text = h_info.get(field, "")
        if text:
            l_t = text.lower().split()
            h_field_cnt[field] += 1
            h_field_len[field] += len(l_t)
            for t in l_t:
                h_field_df[field][t] = h_field_df[field].get(t, 0)

for field in TARGET_TEXT_FIELDS:
    h_field_len[field] /= float(max(h_field_cnt[field], 1))

h_stat = {}
print "start dumping old format..."
for field in TARGET_TEXT_FIELDS:
    h_stat[field]['total_df'] = h_field_cnt[field]
    h_stat[field]['average_len'] = h_field_len[field]
    pickle.dump(h_field_df[field], open(sys.argv[2] + '.' + field, 'wb'))

pickle.dump(h_stat, open(sys.argv[2] + '.stat', 'wb'))
print "finished"



