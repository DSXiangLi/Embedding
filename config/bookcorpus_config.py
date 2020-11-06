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
    'emb_size': 200,
    'cell_size': 1,
    'hidden_units': 100,
    'dtype': tf.float64,
    'max_decoder_iter': 10,
    'start_token': MySpecialToken.SEQ_START,
    'end_token': MySpecialToken.SEQ_END,
    'beam_width': 1,
    'max_len': 20
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)
