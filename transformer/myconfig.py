# -*-coding:utf-8 -*-
import tensorflow as tf
from config.default_config import TRAIN_PARAMS, get_pretrain_model, RUN_CONFIG

WMT_UPDATE = {
    'emb_size': 300,
    'epochs': 20,
    'batch_size': 100,
    'dtype': tf.float32,
    'max_len': 20,
    'min_len': 5,
    'min_count': 3,
    'max_count': 8000,
    'warmup': True,
    'warmup_steps': 10000,
    'pretrain_model_list': [get_pretrain_model('gn300'), get_pretrain_model('ft300')], # [] for random initialization
    'encode_attention_layers': 3,
    'decode_attention_layers': 3,
    'num_head': 12,
    'ffn_hidden': 200,
    'dropout_rate': 0.1,
    'skip_decoder': False
}

ALL_TRAIN_PARAMS = {
    'wmt': dict(TRAIN_PARAMS, **WMT_UPDATE)
}

ALL_RUN_CONFIG = {
    'wmt': RUN_CONFIG
}



