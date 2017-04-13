"""
enumerate possible chunking combinations
input:
    text
    surface form set
output:
    all possible spots combinations (with no overlap)
    in pretty format:
        text \t spot 0 (always the ERD's greedy results)
        text \t spot 1 for this text, with brackets
"""
import logging
import json
max_sf_len = 5


def spot_enumerate(l_terms, s_surface_form):
    ll_grounding = [] # each one is a possible spot
    l_brackets = []
    _dfs_spot(l_terms, s_surface_form, 0, l_brackets, ll_grounding)
    return ll_grounding


def _dfs_spot(l_terms, s_surface_from, st, l_this_bracket, ll_res):
    if st >= len(l_terms):
        if l_this_bracket:
            ll_res.append(l_this_bracket)
            logging.info('get one possible spot enum %s',
                         json.dumps(_bracket_to_spot(l_terms, l_this_bracket)))
        return
    l_e = range(st + 1, min(len(l_terms), st + max_sf_len) + 1)
    l_e.sort(reverse=True)
    for ed in l_e:
        sf = ' '.join(l_terms[st:ed])
        if sf in s_surface_from:
            logging.info('[%d, %d) [%s] spotted', st, ed, sf)
            _dfs_spot(l_terms, s_surface_from, ed, l_this_bracket + [(st, ed)], ll_res)
    _dfs_spot(l_terms, s_surface_from, st + 1, l_this_bracket, ll_res)
    return


def _bracket_to_spot(l_terms, l_bracket):
    l_spot = [' '.join(l_terms[st:ed]) for st, ed in l_bracket]
    return l_spot


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log
    set_basic_log(logging.INFO)

    if 4 != len(sys.argv):
        print "I enumerate query spotting"
        print "3 para: query + surface forms (col 0) + output"
        sys.exit()

    logging.info('loading surface forms')
    s_sf = set([line.strip().split('\t')[0].lower() for line in open(sys.argv[2])])
    logging.info('loaded [%d] sf', len(s_sf))

    out = open(sys.argv[3], 'w')
    l_comb = []
    q_cnt = 0
    for line in open(sys.argv[1]):
        q_cnt += 1
        qid, query = line.strip().split('\t')
        l_qt = query.lower().split()
        ll_grounding = spot_enumerate(l_qt, s_sf)
        logging.info('[%s] has [%d] possible combinations', qid, len(ll_grounding))
        l_comb.append(len(ll_grounding))
        for p, l_brackets in enumerate(ll_grounding):
            l_spot = _bracket_to_spot(l_qt, l_brackets)
            print >> out, qid + '\t%d\t' % p + json.dumps(l_spot)

    out.close()
    logging.info('finished with [%d] q, average [%f] grounding per q',
                 q_cnt, sum(l_comb) / float(q_cnt))



