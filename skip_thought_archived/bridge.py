# -*- coding=utf-8 -*-
"""
Bridge encoder & decoder when state not match, 2 possible mismatch reason
1. hidden size differ(including cell size for rnn encoder)
2. cell type differ: gru vs lstm

Ref from tf-seq2seq: https://github.com/google/seq2seq/blob/master/seq2seq/models/bridges.py
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest


def bridge(encoder_output, decoder_cell):
    with tf.variable_scope('bridging'):
        batch_size = tf.shape(encoder_output.output)[0]
        # 哈哈真的是一点都不能改，本想改成[batch,-1]但是会造成dim1=None,后面Dense的部分会报错
        state = nest.map_structure(lambda x: tf.reshape(x, [batch_size, np.prod(x.get_shape().as_list()[1:])]),
                                   encoder_output.state) # CNN state possible with 3 dimension
        state = nest.flatten(state) # mainly for LSTM, flatten namedtuple to list of c&h
        state = tf.concat(state, 1) # concat multiple cell together

        decoder_shape = nest.flatten(decoder_cell.state_size) # possible LSTMStateTuple into list

        # Do fully connect layer to map the encoder.state and decoder.initial_state
        initial_state = tf.layers.dense(
            inputs = state,
            units = sum(decoder_shape), # to match with decoder total units
            name = 'bridge'
        )

        # split into batch * initial_state(decoder state shape) to enable below pack
        initial_state = tf.split(initial_state, decoder_shape, axis=1)

        # possible packing state into namedtuple like LSTM to be same as decoder hidden state
        return nest.pack_sequence_as(decoder_cell.state_size, initial_state)

