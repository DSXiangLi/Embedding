# -*-coding:utf-8 -*-
import tensorflow as tf
from config.default_config import TRAIN_PARAMS, ModelGN300, RUN_CONFIG

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
    'conditional': True,
    'clip_gradient': True,
    'rate_decay': False,
    'bridge_needed': False, # If encoder & decoder has same cell and shape, turn to False
    'pretrain_model': ModelGN300,
    'skip_decoder': True,
    'decoder_cell': 'gru',
    'encoder_cell': 'gru',
    'encoder_cell_params': RNN_CONFIG['gru'],
    'decoder_cell_params': RNN_CONFIG['gru']
}


ALL_TRAIN_PARAMS = {
    'bookcorpus': dict(TRAIN_PARAMS, **BOOKCORPUS_UPDATE)
}

ALL_RUN_CONFIG = {
    'bookcorpus': RUN_CONFIG
}



