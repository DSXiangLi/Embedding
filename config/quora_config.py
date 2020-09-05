import tensorflow as tf
from config.default_config import  TRAIN_PARAMS, RUN_CONFIG
from utils import get_available_gpus


INVALID_INDEX = 0
ALL_DEVICES = get_available_gpus()
NUM_DEVICES = 2

if len(ALL_DEVICES) > NUM_DEVICES:
    ALL_DEVICES = ALL_DEVICES[:NUM_DEVICES]


QUORA_PROTO = {
    'tokens': tf.VarLenFeature(tf.string),
    'ngram_tokens': tf.VarLenFeature(tf.string),
    'target': tf.FixedLenFeature([], tf.float32),
    'extra_features': tf.FixedLenFeature([3], tf.float32) ## If add new features, don't forget to change this
}

TRAIN_PARAMS_UPDATE = {
    'padded_shape': ({'tokens': [None],
                     'extra_features': [3]}, [1]),
    'padding_values': ({'tokens': INVALID_INDEX, 'extra_features': 0.0}, 0.0),
    'label_size': 1,
    'batch_size': 1000 * NUM_DEVICES,
    'decay_steps': 100000,
    'decay_rate': 0.95,
    'extra_size': 3,
    'emb_size': 200,
    'extra_hidden_size': 2,
    'use_extra': True,
    'ngram': 1,
    'invalid_index': INVALID_INDEX
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)


RUN_CONFIG_UPDATE = {
    'devices': ALL_DEVICES,
    'summary_steps': 200,
    'log_steps': 200,
    'keep_checkpoint_max':1,
    'save_steps': 200
}


RUN_CONFIG.update(RUN_CONFIG_UPDATE)
