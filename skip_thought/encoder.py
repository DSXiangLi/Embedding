# -*- coding=utf-8 -*-
"""

Encoder Collectionn

"""
import tensorflow as tf
from collections import namedtuple

from skip_thought.seq2seq_utils import encoder_decoder_collection, build_rnn_cell

ENCODER_OUTPUT = namedtuple('EncoderOutput', ['output', 'state'])


@encoder_decoder_collection('gru_encoder')
def gru_encoder(input_emb, input_len, params):
    gru_cell = build_rnn_cell('gru', params)

    # state: batch_size * hidden_size, output: batch_size * max_len * hidden_size
    output, state = tf.nn.dynamic_rnn(
        cell=gru_cell, # one rnn units
        inputs=input_emb, # batch_size * max_len * feature_size
        sequence_length=input_len, # batch_size * seq_len
        initial_state=None,
        dtype=params['dtype'],
        time_major=False # whether reshape max_length to first dim
    )
    return ENCODER_OUTPUT(output=output, state=state)


