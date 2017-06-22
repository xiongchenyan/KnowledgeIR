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
from keras.models import Model
from keras.layers import (
    Embedding,
    dot,
    Input
)


def cos_model(emb_mtx):
    m = Embedding(
        emb_mtx.shape[0],
        emb_mtx.shape[1],
        weights=[emb_mtx],
        name="embedding",
        trainable=False,
    )
    q_in = Input(shape=(None,), name='q')
    d_in = Input(shape=(None,), name='d')
    q = m(q_in)
    d = m(d_in)
    cos = dot([q, d], axes=-1, normalize=True, name='translation_mtx')
    model = Model(inputs=[q_in, d_in], outputs=cos)
    return model


def batch_cos(q_batch, d_batch):
    cos = K.batch_dot(K.l2_normalize(q_batch, -1), K.l2_normalize(d_batch, -1), [2, 2]).eval()
    return cos


def compute_translation_mtx(in_dir, calc_cos_model, aux=False):
    q = np.load(os.path.join(in_dir, 'q.npy'))
    logging.info('q shape %s', json.dumps(q.shape))

    l_name = ['d_title.npy', 'd_bodyText.npy']

    for name in l_name:
        fname = os.path.join(in_dir, name)
        d = np.load(fname)
        logging.info('[%s] shape %s', name, json.dumps(d.shape))
        cos = calc_cos_model.predict([q, d])
        logging.info('cos shape %s', json.dumps(cos.shape))
        np.save(os.path.join(in_dir, 'translation_mtx_' + name), cos)

    if aux:
        for name in l_name:
            fname = os.path.join(in_dir, 'aux_' + name)
            d = np.load(fname)
            logging.info('[%s] shape %s', name, json.dumps(d.shape))
            cos = calc_cos_model.predict([q, d])
            logging.info('cos shape %s', json.dumps(cos.shape))
            np.save(os.path.join(in_dir, 'aux_translation_mtx_' + name), cos)
    logging.info('[%s] finished')
    return

if __name__ == '__main__':
    import sys
    if 3 != len(sys.argv):
        print "pre compute translation matrix"
        print "2 para: the processed npy data folder + embeding npy"
        sys.exit(-1)

    emb_mtx = np.load(sys.argv[2])
    calc_cos_model = cos_model(emb_mtx)
    compute_translation_mtx(os.path.join(sys.argv[1], 'pairwise'), calc_cos_model, True)
    compute_translation_mtx(os.path.join(sys.argv[1], 'pointwise'), calc_cos_model, False)


