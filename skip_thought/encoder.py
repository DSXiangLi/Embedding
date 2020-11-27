# -*- coding=utf-8 -*-
"""
Encoder Collectionn
    1. gru encoder
    2. cnn encoder

"""
import tensorflow as tf
from collections import namedtuple

from skip_thought.seq2seq_utils import encoder_decoder_collection, build_rnn_cell

ENCODER_OUTPUT = namedtuple('EncoderOutput', ['output', 'state'])


@encoder_decoder_collection('gru_encoder')
def gru_encoder(input_emb, input_len, params):
    cell = build_rnn_cell(params['encoder_cell'], params['encoder_cell_params'])

    # state: batch_size * hidden_size, output: batch_size * max_len * hidden_size
    output, state = tf.nn.dynamic_rnn(
        cell=cell, # one rnn units
        inputs=input_emb, # batch_size * max_len * feature_size
        sequence_length=input_len, # batch_size * seq_len
        initial_state=None,
        dtype=params['dtype'],
        time_major=False # whether reshape max_length to first dim
    )
    return ENCODER_OUTPUT(output=output, state=state)


@encoder_decoder_collection('cnn_encoder')
def cnn_encoder(input_emb, input_len, params):
    # batch_szie * seq_len * emb_size -> batch_size * (seq_len-kernel_size + 1) * filters
    outputs = []
    params = params['encoder_cell_params']
    for i in range(len(params['filters'])):
        output = tf.layers.conv1d(inputs = input_emb,
                                  filters = params['filters'][i],
                                  kernel_size = params['kernel_size'][i], # window size, simlar as n-gram
                                  strides = params['strides'][i],
                                  padding = params['padding'][i]
                                )
        output = params['activation'][i](output)
        ## TODO: dropout, but dropout in convolution is questionable. BN maybe?
        # batch_size * (seq_len-kernel_size + 1) * filters -> batch_size * filters
        outputs.append(tf.reduce_max(output, axis=1))
    # batch_size * sum(filters)
    output = tf.concat(outputs, axis=1)
    return ENCODER_OUTPUT(output=output, state=(output,))


