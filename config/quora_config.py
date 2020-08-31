import tensorflow as tf
from config.default_config import INVALID_INDEX, TRAIN_PARAMS

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
    'batch_size': 1000,
    'decay_steps': 100000,
    'decay_rate': 0.95,
    'extra_size': 3,
    'emb_size': 256,
    'extra_hidden_size': 2,
    'use_extra': True,
    'ngram': 1
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)
