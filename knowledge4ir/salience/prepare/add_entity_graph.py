"""
Add entity-event graph to docs.
"""
import sys
import json
import gzip

if not len(sys.argv) == 4:
    print("Usage: json_align.py [original file] [adjacent file] [output file]")


def zopen(fname):
    if fname.endswith('gz'):
        return gzip.open(fname, 'rt')
    else:
        return open(fname)


loaded_lines = {}
with zopen(sys.argv[2]) as adj_file:
    line_num = 0
    for line in adj_file:
        doc = json.loads(line)
        docno = doc['docno']
        loaded_lines[docno] = line
        line_num += 1
        sys.stdout.write("Loaded %d input lines.\r" % line_num)
print("\nDone loading input files.")

with zopen(sys.argv[1]) as ee_file, open(sys.argv[3], 'w') as output_file:
    line_num = 0
    missed = 0

    for line in ee_file:
        line_num += 1
        doc = json.loads(line)
        docno = doc['docno']

        adj_line = None

        try:
            adj_line = loaded_lines[docno]
        except KeyError as e:
            missed += 1

        if adj_line:
            adj_info = json.loads(adj_line)
            doc['adjacentList'] = adj_info['adjacentList']

        output_file.write(json.dumps(doc))
        output_file.write("\n")

        sys.stdout.write("Processed %d ordered lines, missed %d lines.\r" % (
            line_num, missed))
