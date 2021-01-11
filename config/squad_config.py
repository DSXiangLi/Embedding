# -*- coding=utf-8 -*-
import tensorflow as tf
from config.default_config import SpecialSeqToken, RUN_CONFIG, TRAIN_PARAMS, ModelGN300

MySpecialToken = SpecialSeqToken(SEQ_START='<GO>',
                                 SEQ_END='<EOS>',
                                 PAD='<PAD>',
                                 UNK='<UNK>')

TRAIN_PARAMS_UPDATE = {
    'emb_size': 300,
    'epochs': 20,
    'dtype': tf.float32,
    'max_decode_iter': 10,
    'min_len': 1,
    'beam_width': 1,
    'min_count': 2,
    'max_count': 50000,
    'batch_size': 50,
    'learning_rate': 0.001,
    'clip_gradient': False,
    'rate_decay': False,
    'pretrain_model': ModelGN300,
    'encode_attention_layers': 4,
    'decode_attention_layers': 4,
    'num_head': 6,
    'dropout_rate': 0.1,
    'skip_decoder': False
}


TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)

