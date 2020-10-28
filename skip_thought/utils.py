# -*- coding=utf-8 -*-

import tensorflow as tf
from collections import namedtuple

SEQ_LOSS_OUTPUT = namedtuple('SEQ_LOSS_OUTPUT', ['loss_id', 'loss_per_sample', 'loss_per_time', 'loss_per_batch'])
SEQ_PRED_OUTPUT = namedtuple('SEQ_PRED_OUTPUT', ['predict_prob', 'predict_id', 'predict_tokens', 'seq_len'])
ENCODER_FAMILY = {}
DECODER_FAMILY = {}


def encoder_decoder_collection(func_name):
    def helper(func):
        if 'encoder' in func_name:
            ENCODER_FAMILY[func_name] = func
        elif 'decoder' in func_name:
            DECODER_FAMILY[func_name] = func
        else:
            raise Exception('Only encoder and decoder are supported in func_name')
        return func
    return helper


def build_rnn_cell(cell_type, params):
    if cell_type.lower() == 'rnn':
        cell_class = tf.nn.rnn_cell.RNNCell
    elif cell_type.lower() == 'gru':
        cell_class = tf.nn.rnn_cell.GRUCell
    elif cell_type.lower() == 'lstm':
        cell_class = tf.nn.rnn_cell.LSTMCell
    else:
        raise Exception('Only rnn, gru, lstm are supported as cell_type')

    return tf.nn.rnn_cell.MultiRNNCell(
        cells = [ cell_class(num_units = params['hidden_units'][i]) for i in range(params['cell_size']) ]
    )


def embedding_func(embedding):
    def helper(id):
        return tf.nn.embedding_lookup(embedding, id)
    return helper


def model_fn(model):
    def helper(features, labels, params, mode):
        model_ = model(params)
        return model_.build_model(features, labels, mode)
    return helper
