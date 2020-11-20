# -*- coding=utf-8 -*-
import tensorflow as tf
from config.default_config import SpecialSeqToken, RUN_CONFIG, TRAIN_PARAMS

MySpecialToken = SpecialSeqToken(SEQ_START = '<GO>',
                                 SEQ_END = '<EOS>',
                                 PAD = '<PAD>',
                                 UNK = '<UNK>')

TRAIN_PARAMS_UPDATE = {
    'encoder_type': 'gru_encoder',
    'decoder_type': 'gru_decoder',
    'hidden_units': [100],
    'emb_size': 300,
    'cell_size': 1,
    'dtype': tf.float32,
    'max_decode_iter': 8,
    'beam_width': 1,
    'min_count': 5,
    'max_count': 10000,
    'batch_size': 20,
    'learning_rate': 0.001,
    'conditional': True,
    'clip_gradient': True,
    'rate_decay': False
}


TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)
