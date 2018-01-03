import logging
import math
import json


def bin_score(l_stat, l_score, nb_bin):
    l_item = zip(l_stat, l_score)
    l_item.sort(key=lambda item: item[0])
    l_sorted_score = [item[1] for item in l_item]
    bin_width = int(math.ceil(len(l_stat) / float(nb_bin)))
    logging.info('binning [%d] data to [%d] size [%d] bin', len(l_item), nb_bin, bin_width)
    # logging.info('[%d] item: %s', len(l_item), json.dumps(l_item))
    l_bin_res = []
    st = 0
    ed = bin_width
    l_bin_range = []
    while st < len(l_sorted_score):
        l_this_bin = l_sorted_score[st: ed]
        logging.info('bin [%d] [%d:%d]', len(l_bin_res), st, ed)
        b_st, b_ed = l_item[st][0], l_item[min(ed, len(l_item)-1)][0]
        logging.info('bin range %f, %f', b_st, b_ed)
        l_bin_range.append((b_st, b_ed))
        score = sum(l_this_bin) / float(len(l_this_bin))
        st = ed
        ed += bin_width
        l_bin_res.append(score)
    return l_bin_res, l_bin_range
