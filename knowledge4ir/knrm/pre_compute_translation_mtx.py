"""
pre compute the translation matrix from npy converted data
input:
    the folder with output from data_reader.py
    folder/
        pairwise/
            q.npy d_field.npy aux_d_field.npy
        pointwise/
            q.npy d_field.npy
    an embedding mtx.npy
do:
    pre-compute the translation matrix between
        q-d_field, q_aux_d_field
output:
    put in the same folder:
        translation_mtx_d_field.npy
        aux_translation_mtx_d_field.npy
"""

import json
import os
import logging
import numpy as np
import keras.backend as K


def batch_cos(q_batch, d_batch):
    cos = K.batch_dot(K.l2_normalize(q_batch, -1), K.l2_normalize(d_batch, -1), [2, 2]).eval()
    return cos


def compute_translation_mtx(in_dir, aux=False):
    q = np.load('q.npy')
    logging.info('q shape %s', json.dumps(q.shape))

    l_name = ['d_title.npy', 'd_bodyText.npy']

    for name in l_name:
        fname = os.path.join(in_dir, name)
        d = np.load(fname)
        logging.info('[%s] shape %s', name, json.dumps(d.shape))
        cos = batch_cos(q, d)
        logging.info('cos shape %s', json.dumps(cos.shape))
        np.save(os.path.join(in_dir, 'translation_mtx_' + name), cos)

    if aux:
        for name in l_name:
            fname = os.path.join(in_dir, 'aux_' + name)
            d = np.load(fname)
            logging.info('[%s] shape %s', name, json.dumps(d.shape))
            cos = batch_cos(q, d)
            logging.info('cos shape %s', json.dumps(cos.shape))
            np.save(os.path.join(in_dir, 'aux_translation_mtx_' + name), cos)
    logging.info('[%s] finished')
    return

if __name__ == '__main__':
    import sys
    if 2 != len(sys.argv):
        print "pre compute translation matrix"
        print "1 para: the processed npy data folder"
        sys.exit(-1)
    compute_translation_mtx(os.path.join(sys.argv[1], 'pairwise'), True)
    compute_translation_mtx(os.path.join(sys.argv[1], 'pointwise'), False)


