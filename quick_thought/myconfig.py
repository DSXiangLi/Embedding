# -*-coding:utf-8 -*-
import tensorflow as tf
from config.default_config import TRAIN_PARAMS, get_pretrain_model, RUN_CONFIG

RNN_CONFIG = {
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
    'clip_gradient': True,
    'pretrain_model_list': [get_pretrain_model('gn300')],
    'skip_decoder': True, # For quick thought, in predict mode no decoder is needed
    'window_size': 1, # positive label range
    'encoder_cell': 'gru',
    'encoder_cell_params': RNN_CONFIG['gru']
}


ALL_TRAIN_PARAMS = {
    'bookcorpus': dict(TRAIN_PARAMS, **BOOKCORPUS_UPDATE)
}

ALL_RUN_CONFIG = {
    'bookcorpus': RUN_CONFIG
}



