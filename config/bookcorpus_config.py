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
    'hidden_units': [100], ## must be the length of cell_size
    'emb_size': 200,
    'cell_size': 1,
    'dtype': tf.float32,
    'max_decode_iter': 20,
    'beam_width': 1,
    'batch_size':10
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)
