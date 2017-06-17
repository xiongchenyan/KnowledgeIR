"""
convert boe representation (with attention) to tensor
input:
    boe info
    embedding (Use as entity id hash)
output:
    each line:
        boe id vector
        attention feature matrix
    and a numpy embedding
"""

import numpy as np
import json
import sys
import logging


def load_embedding(word2vec_in):
    l_e = []
    l_emb = []

    logging.info('start loading embedding')
    for p, line in enumerate(open(word2vec_in)):
        if not p:
            continue
        if not p % 1000:
            logging.info('loaded %d lines')

