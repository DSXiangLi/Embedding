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
    'max_decode_iter': 8,
    'beam_width': 1,
    'min_count': 2,
    'max_count': 50000,
    'batch_size': 20,
    'learning_rate': 0.001,
    'conditional': True,
    'clip_gradient': True,
    'rate_decay': False,
    'bridge_needed': True # If encoder & decoder has same cell and shape, turn to False
}


TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)

# Encoder，decoder must have same state shape, otherwise bridge is needed
ED_PARAMS = {
    'gru': {
        'hidden_units': [100],
        'cell_size': 1,
        'keep_prob': [0.9]
    },
    'cnn': {
        'filters': [50] * 2,
        'kernel_size': [3, 4],
        'strides': [1] * 2,
        'padding': ['VALID'] * 2,
        'activation': [tf.nn.tanh] * 2,
        'pooling': [tf.nn.max_pool1d] * 2,
        'dropout_rate': [0.1, 0.1, 0.1]
    },
    'lstm': {
        'hidden_units': [90],
        'cell_size': 1,
        'dropout_rate': [0.1]
    }
}


