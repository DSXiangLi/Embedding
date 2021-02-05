# -*-coding:utf-8 -*-
import tensorflow as tf
from config.default_config import TRAIN_PARAMS, get_pretrain_model, RUN_CONFIG

RNN_CONFIG = {
    'cnn': {
        'filters': [50] * 3,
        'kernel_size': [2, 3, 4 ], # max kernel size must be smaller than min_len
        'strides': [1] * 3,
        'padding': ['VALID'] * 3,
        'activation': [tf.nn.tanh] * 3,
        'pooling': [tf.nn.max_pool1d] * 3,
        'keep_prob': [0.9, 0.9, 0.9]
    },
    'lstm': {
        'hidden_units': [90],
        'cell_size': 1,
        'keep_prob': [0.9]
    },
    'gru': {
        'hidden_units': [150],
        'cell_size': 1,
        'keep_prob': [0.9]
    }
}

BOOKCORPUS_UPDATE = {
    'emb_size': 300,
    'epochs': 20,
    'batch_size': 50,
    'dtype': tf.float32,
    'min_len': 4,
    'max_len': 10,
    'min_count': 2,
    'max_count': 50000,
    'learning_rate': 0.001,
    'conditional': True,
    'clip_gradient': True,
    'rate_decay': False,
    'bridge_needed': True, # If encoder & decoder has same cell and shape, turn to False
    'pretrain_model_list': [get_pretrain_model('gn300')],
    'skip_decoder': True,
    'decoder_cell': 'gru',
    'encoder_cell': 'cnn',
    'encoder_cell_params': RNN_CONFIG['cnn'],
    'decoder_cell_params': RNN_CONFIG['gru']
}


ALL_TRAIN_PARAMS = {
    'bookcorpus': dict(TRAIN_PARAMS, **BOOKCORPUS_UPDATE)
}

ALL_RUN_CONFIG = {
    'bookcorpus': RUN_CONFIG
}



