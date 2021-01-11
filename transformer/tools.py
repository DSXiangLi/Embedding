# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np

from encoder_decodeer_helper.tools import seq_mask_gen


def layer_norm(x):
    """
    layer normalization from Jimmy, apply normalization along feature and apply transformation
    """
    with tf.variable_scope('layer_normalization', reuse=tf.AUTO_REUSE):
        epsilon = tf.constant(np.finfo(np.float32).eps)
        mean, variance = tf.nn.moments(x, axes=-1, keep_dims=True)
        x = (x - mean)/((variance + epsilon)**0.5)
        x = tf.layers.dense(x, units=x.get_shape().as_list()[-1],
                            activation=None, use_bias=True)
    return x


def add_and_norm_layer(x, sub_layer_x):
    """
    combine Residual connection & layer_norm
    """
    with tf.variable_scope('add_and_norm'):
        x = tf.add(x, sub_layer_x)
        x = layer_norm(x)

    return x


def ffn(x, params, mode):
    """
    feed forward after add & norm
    """
    with tf.variable_scope('ffn', reuse=tf.AUTO_REUSE):
        d_model = x.shape.as_list()[-1]  # emb_size
        y = tf.layers.dense(x, units=d_model, activation='relu')
        y = tf.layers.dense(y, units=d_model, activation=None)
        y = tf.layers.dropout(y, rate=params['dropout_rate'],
                              training=(mode == tf.estimator.ModeKeys.TRAIN))

        y = add_and_norm_layer(x, y)

    return y


def future_mask_gen(input_, params):
    """
    In decoder self-attention, additional future_mask is needed to avoid leakage. add future mask on padding mask
    input:
        input_: {'tokens':, 'seq_len'}, seq_len is non-padded sequence len
    output:
        mask:  batch_size * input_len * input_len, with 1 to keep, 0 to drop
    """
    # give 0 to all padding position for both key and query
    seq_mask = seq_mask_gen(input_, params)
    # batch_size * key_len * key_len(seq_len)
    mask = tf.matmul(seq_mask, seq_mask, transpose_b=True)
    # keep upper triangle with diagonal
    mask = tf.matrix_band_part(mask, num_lower=0, num_upper=-1)

    return mask


def scaled_dot_product_attention(key, value, query, mask):
    """
    apply dot product attention with mask
    input:
        key: batch_size * key_len * emb_size
        query: batch_size * query_len * emb_size
        value: batch_size * key_len * emb_size
        mask: batch_size * key_len
    output:
        weighted_val: batch_size * query_len * emb_size
    """
    with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
        # scalaed weight matrix : batch_size * query_len * key_len
        dk = key.shape.as_list()[-1] # emb_size
        weight = tf.matmul(query, tf.transpose(key, [0, 2, 1]))/(dk**0.5)

        # apply mask: large negative will become 0 in softmax
        weight += (1-mask) * (-2**32+1)

        # normalize on axis key_len so that score add up to 1
        weight = tf.nn.softmax(weight, axis=-1)

        # weighted value: batch_size * query_len * emb_size
        weighted_value = tf.matmul(weight, value )

        return weighted_value


def multi_head_attention(key, value, query, mask, params, mode):
    """
    Mutlihead attention with mask
    input:
        key: batch_size * key_len * emb_size
        query: batch_size * query_len * emb_size
        value: batch_size * key_len * emb_size
        mask: batch_size * key_len
    output:
        weighted_val: batch_size * query_len * emb_size
    """
    ##TODO: figure out the different between multihead and
    with tf.variable_scope('multi_head_attention', reuse=tf.AUTO_REUSE):
        d_model = value.shape.as_list()[-1] # emb_size
        # linear projection with dimension unchaangned
        new_key = tf.layers.dense(key, units=d_model, activation=None) # batch_size * key_len * emb_size
        new_value = tf.layers.dense(value, units=d_model, activation=None)
        new_query = tf.layers.dense(query, units=d_model, activation=None)

        # split d_model by num_head and compute attention in parallel
        # (batch_size * num_head) * key_len * (emb_size/num_head)
        new_key = tf.concat(tf.split(new_key, num_or_size_splits=params['num_head'], axis=-1), axis=0)
        new_value = tf.concat(tf.split(new_value, num_or_size_splits=params['num_head'], axis=-1), axis=0)
        new_query = tf.concat(tf.split(new_query, num_or_size_splits=params['num_head'], axis=-1), axis=0)

        # calculate dot-product attention
        weighted_val = scaled_dot_product_attention(new_key, new_value, new_query, tf.tile(mask, [params['num_head'], 1, 1]))

        # concat num_head back
        # (batch_size * num_head) * query_len * (emb_size/num_head) -> batch_size * query_len * emb_size
        weighted_val = tf.concat(tf.split(weighted_val, num_or_size_splits=params['num_head'], axis=0), axis=-1)

        # Linear projection
        weighted_val = tf.layers.dense(weighted_val, units=d_model, activation=None)
        # Do dropout
        weighted_val = tf.layers.dropout(weighted_val, rate=params['dropout_rate'],
                                         training=(mode == tf.estimator.ModeKeys.TRAIN))

        weighted_val = add_and_norm_layer(query, weighted_val)

    return weighted_val


def positional_encoding(x):
    """
    inject relative and absolute position information
    inputs:
        x: batch_size * pad_len * emb_size
    output:
        encoding: pad_lenn * emb_size
    """
    with tf.variable_scope('positional_encoding', reuse=tf.AUTO_REUSE):
        d_model = x.shape.as_list()[-1]
        seq_len = x.shape.as_list()[-2]

        encoding_row = np.array([10000**((i//2)/d_model) for i in range(d_model)])
        encoding_matrix = np.array([i/encoding_row for i in range(seq_len)])

        def sin_cos(row):
            row = [np.cos(val) if i%2 else np.sin(val) for i, val in enumerate(row)]
            return row

        encoding_matrix = np.apply_along_axis(sin_cos, 1, encoding_matrix)
        encoding_matrix = tf.constant(encoding_matrix)

    return encoding_matrix


