import tensorflow as tf
from config.default_config import  TRAIN_PARAMS, RUN_CONFIG, MyWordSpecialToken


MySpecialToken = MyWordSpecialToken


TF_PROTO = {
    'tokens': tf.VarLenFeature(tf.string),
    'ngram_tokens': tf.VarLenFeature(tf.string),
    'target': tf.FixedLenFeature([], tf.float32),
    'extra_features': tf.FixedLenFeature([3], tf.float32) ## If add new features, don't forget to change this
}

TRAIN_PARAMS_UPDATE = {
    'label_size': 1,
    'decay_steps': 100000,
    'decay_rate': 0.95,
    'extra_size': 3,
    'emb_size': 200,
    'extra_hidden_size': 2,
    'use_extra': True,
    'ngram': 1
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)

RUN_CONFIG_UPDATE = {
    'summary_steps': 200,
    'log_steps': 200,
    'keep_checkpoint_max':1,
    'save_steps': 200
}

RUN_CONFIG.update(RUN_CONFIG_UPDATE)
