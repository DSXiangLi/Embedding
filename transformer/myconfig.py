# -*-coding:utf-8 -*-
import tensorflow as tf
from config.default_config import TRAIN_PARAMS, ModelGN300, RUN_CONFIG

BOOKCORPUS_UPDATE = {
    'emb_size': 300,
    'epochs': 20,
    'dtype': tf.float32,
    'max_len': 10,
    'min_len': 4,
    'min_count': 2,
    'max_count': 50000,
    'batch_size': 50,
    'learning_rate': 0.001,
    'clip_gradient': False,
    'rate_decay': False,
    'warmup': True,
    'warmup_steps': 10000,
    'pretrain_model': ModelGN300,
    'encode_attention_layers': 6,
    'decode_attention_layers': 6,
    'num_head': 6,
    'ffn_hidden': 300,
    'dropout_rate': 0.1,
    'skip_decoder': False
}

ALL_TRAIN_PARAMS = {
    'bookcorpus': dict(TRAIN_PARAMS, **BOOKCORPUS_UPDATE)
}

ALL_RUN_CONFIG = {
    'bookcorpus': RUN_CONFIG
}



