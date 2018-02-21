"""
basic functions of result analysis
"""
import random


def randomization_test(l_target, l_base):
    total_test = 1000
    diff = sum(l_target) / float(len(l_target)) - sum(l_base) / float(len(l_base))
    cnt = 0.0
    for i in range(total_test):
        l_a, l_b = random_swap(l_target, l_base)
        this_diff = sum(l_a) / float(len(l_a)) - sum(l_b) / float(len(l_b))
        if this_diff > diff:
            cnt += 1.0
    p = cnt / float(total_test)
    return p


def random_swap(l_target, l_base):
    l_a = list(l_target)
    l_b = list(l_base)

    for i in range(len(l_target)):
        if random.randint(0, 1):
            l_a[i], l_b[i] = l_b[i],l_a[i]
    return l_a, l_b


def win_tie_loss(l_target, l_base):
    l_a = [round(a, 3) for a in l_target]
    l_b = [round(b, 3) for b in l_base]
    l_ab = zip(l_a, l_b)
    win = sum([int(a > b) for a, b in l_ab])
    tie = sum([int(a == b) for a, b in l_ab])
    loss = sum([int(a < b) for a, b in l_ab])

    return win, tie, loss
