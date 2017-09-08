"""
prepare corpus stats
field -> h_df
field -> total df
field -> avg len

json dump to a list []

convert old version of corpus_stat to the new version in utils
"""

import sys
from knowledge4ir.utils import (
    title_field,
    body_field
)
import pickle
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


if 3 != len(sys.argv):
    print "convert old version of corpus_stat to the new version, which is packed in one class"
    print "2 paras: input pre + out"
    sys.exit(-1)

print "start loading pickle dicts..."
h_title_df = pickle.load(open(sys.argv[1] + '.title'))
print "[%d] title term" % len(h_title_df)
h_body_df = pickle.load(open(sys.argv[1] + '.bodyText'))
print '[%d] body term' % len(h_body_df)
h_stat = pickle.load(open(sys.argv[1] + '.stat'))

print 'dicts loaded'

h_field_df = dict()
h_field_df[title_field] = h_title_df
h_field_df[body_field] = h_body_df

h_field_total_df = dict()
h_field_total_df[title_field] = h_stat[title_field]['total_df']
h_field_total_df[body_field] = h_stat[body_field]['total_df']

h_field_avg_len = dict()
h_field_avg_len[title_field] = h_stat[title_field]['average_len']
h_field_avg_len[body_field] = h_stat[body_field]['average_len']

l_data = [h_field_df, h_field_total_df, h_field_avg_len]
print "start dumping new format..."
pickle.dump(l_data, open(sys.argv[2], 'wb'))
print "finished"

