"""
add e name to each line
input:
    line with e id
    col of e id in the line
    e id -> name in (columns)
output:
    add e name in the end of each line
"""


def add_e_name(line, h_e_id, col_p):
    cols = line.strip().split()
    name = h_e_id.get(cols[col_p], 'NA')
    return '\t'.join(cols + [name])


def process(in_name, name_dict_in, out_name, col_p):
    l_cols = [line.strip().split('\t') for line in open(name_dict_in)]
    l_cols = [col for col in l_cols if len(col) == 2]
    h_e_id = dict(l_cols)
    out = open(out_name, 'w')
    for p, line in enumerate(open(in_name)):
        if not p % 1000:
            print "Added [%d] line" % p

        print >> out, add_e_name(line, h_e_id, col_p)
    out.close()


if __name__ == '__main__':
    import sys
    if 5 != len(sys.argv):
        print "add e name to lines"
        print "4 para: to add name file + name id mapping + out name + col_p"
        sys.exit(-1)

    col_p = int(sys.argv[4])
    in_name, name_dict_in, out_name = sys.argv[1:4]
    process(in_name, name_dict_in, out_name, col_p)

