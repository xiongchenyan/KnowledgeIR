"""
Align json file according to existing order.
"""
from __future__ import print_function
import sys
import json
import os

if not len(sys.argv) == 4:
    print("Usage: json_align.py [ordered file] [input file] [output file]")

ordering = []
with open(sys.argv[1]) as order_file:
    line_num = 0
    for line in order_file:
        doc = json.loads(line)
        docno = doc['docno']
        ordering.append(docno)
        line_num += 1
        sys.stdout.write("Loaded %d orders.\r" % line_num)
print("\nDone loading orderings.")

loaded_lines = {}
with open(sys.argv[2]) as input_file:
    line_num = 0
    for line in input_file:
        doc = json.loads(line)
        docno = doc['docno']
        loaded_lines[docno] = line
        line_num += 1
        sys.stdout.write("Loaded %d input lines.\r" % line_num)
print("\nDone loading input files.")

with open(sys.argv[3], 'w') as output_file:
    line_num = 0
    missed = 0
    for docno in ordering:
        line_num += 1
        try:
            line = loaded_lines[docno]
            output_file.write(line)
        except KeyError as e:
            missed += 1

        sys.stdout.write("Processed %d orderred lines, missed %d lines.\r" % (
            line_num, missed))
print("\nDone reordering.")
