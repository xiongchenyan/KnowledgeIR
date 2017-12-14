#!/usr/bin/env bash
input=$1
emb_out=$2
qsub "./word2vec -train ${input} -output ${emb_out} -size 128 -window 20 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -threads 16 -min-count 5 --save-vocab ${emb_out}.vocab"
