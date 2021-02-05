# -*- coding=utf-8 -*-
import tensorflow as tf
from config.default_config import SpecialSeqToken, RUN_CONFIG, TRAIN_PARAMS

MySpecialToken = SpecialSeqToken(SEQ_START = '<GO>',
                                 SEQ_END = '<EOS>',
                                 PAD = '<PAD>',
                                 UNK = '<UNK>')

TRAIN_PARAMS_UPDATE = {
    'emb_size': 300,
    'epochs': 20,
    'dtype': tf.float32,
    'max_len': 10,
    'min_len': 4,
    'min_count': 2,
    'max_count': 50000,
    'batch_size': 100,
    'learning_rate': 0.001
}


TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)
