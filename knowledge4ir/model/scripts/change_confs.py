"""
change multiple confs
input:
    names
will rewrite those files
"""

import sys


def change_file(in_name):
    lines = open(in_name).read().splitlines()
    for i in xrange(len(lines)):
        if "c.CrossValidator.nb_repeat=" in lines[i]:
            lines[i] = "c.CrossValidator.nb_repeat=20"
    print >> open(in_name, 'w'), '\n'.join(lines)


for in_name in open(sys.argv[1]):
    in_name = in_name.strip()
    change_file(in_name)
