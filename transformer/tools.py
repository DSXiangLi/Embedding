# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np

from encoder_decodeer_helper.tools import seq_mask_gen
from utils import add_layer_summary


def layer_norm(x):
    """
    layer normalization from Jimmy, apply normalization along feature and apply transformation
    """
    with tf.variable_scope('layer_normalization', reuse=tf.AUTO_REUSE):
        d_model = x.shape.as_list()[-1]
        epsilon = tf.constant(np.finfo(np.float32).eps)
        mean, variance = tf.nn.moments(x, axes=-1, keep_dims=True)
        x = (x - mean)/((variance + epsilon)**0.5) # do layer norm
        add_layer_summary('norm', x)

        kernel = tf.get_variable('norm_kernel', shape=(d_model,), initializer=tf.ones_initializer())
        bias = tf.get_variable('norm_bias', shape=(d_model,),initializer=tf.zeros_initializer())
        x= tf.multiply(kernel, x) +bias
        add_layer_summary('norm_transform', x)
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
        y = tf.layers.dense(x, units=params['ffn_hidden'], activation='relu')

        add_layer_summary('ffn_hidden1', y)
        y = tf.layers.dense(y, units=d_model, activation=None)
        y = tf.layers.dropout(y, rate=params['dropout_rate'],
                              training=(mode == tf.estimator.ModeKeys.TRAIN))
        add_layer_summary('ffn_hidden2', y)
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
    seq_mask = seq_mask_gen(input_, params) # batch_size * 1 * key_len
    # batch_size * key_len * key_len(seq_len)
    mask = tf.matmul(seq_mask, seq_mask, transpose_a=True)
    # keep lower triangle with diagonal
    mask = tf.matrix_band_part(mask, num_lower=-1, num_upper=0)

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
        dk = tf.cast(key.shape.as_list()[-1], tf.float32)# emb_size
        weight = tf.matmul(query, key, transpose_b=True)/(dk**0.5)

        # apply mask: large negative will become 0 in softmax[mask=0 ignore]
        weight += (1-mask) * (-2**32+1)
        # normalize on axis key_len so that score add up to 1
        weight = tf.nn.softmax(weight, axis=-1)
        tf.summary.image("attention", tf.expand_dims(weight[:1], -1))  # add channel dim
        add_layer_summary('attention', weight)
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
        add_layer_summary('raw_multi_head', weighted_val)
        weighted_val = add_and_norm_layer(query, weighted_val)

    return weighted_val


def positional_encoding(d_model, max_len, dtype):
    """
    inject relative and absolute position information
    inputs:
        x: batch_size * pad_len * emb_size
    output:
        encoding: max_len * emb_size
    """
    with tf.variable_scope('positional_encoding'):
        encoding_row = np.array([10000**((i-i%2)/d_model) for i in range(d_model)])
        encoding_matrix = np.array([i/encoding_row for i in range(max_len)])

        def sin_cos(row):
            row = [np.cos(val) if i%2 else np.sin(val) for i, val in enumerate(row)]
            return row

        encoding_matrix = np.apply_along_axis(sin_cos, 1, encoding_matrix)
        encoding_matrix = tf.cast(tf.constant(encoding_matrix), dtype)

    return encoding_matrix


def init_input_embedding(embedding, pos_encoding, params):
    def helper(tokens, mode):
        batch_size = tf.shape(tokens)[0]
        pad_len = tf.shape(tokens)[1]
        pos_id = tf.tile(tf.expand_dims(tf.range(pad_len), 0), [batch_size, 1]) # batch_size * padded_len

        we = tf.nn.embedding_lookup(embedding, tokens)
        pe = tf.nn.embedding_lookup(pos_encoding, pos_id)
        add_layer_summary('raw_word_embedding', we)
        add_layer_summary('raw_positional_encoding', pe)

        seq_emb_input = tf.layers.dropout(we+pe, rate=params['dropout_rate'],
                                          training=(mode == tf.estimator.ModeKeys.TRAIN))
        add_layer_summary('embedding_input', seq_emb_input)
        return seq_emb_input
    return helper


if __name__ == '__main__':
    import tensorflow as tf
    sess = tf.Session()

    ## visualize positional encoding
    PE = positional_encoding(d_model=100, max_len=10, dtype=tf.float32)
    PE = sess.run(PE)
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.heatmap(PE, cmap='coolwarm')
    ax.set_xlabel('d_model')
    ax.set_ylabel('max_len')
    plt.show()

    # check PE + embedding
    func = init_input_embedding(PE, PE, {'dtype':tf.float32, 'dropout_rate':0.1})
    func(tf.constant([[1,2,3,4], [2,3,4,5]]), tf.estimator.ModeKeys.TRAIN)

    # check mask
    features = {'seq_len':[3,4,5], 'tokens':[[1,2,3,-1,-1],[2,3,4,5,-1],[3,4,5,6,7]]}
    params = {'dtype':tf.float32}
    seq_mask =seq_mask_gen(features, params)
    sess.run(seq_mask)
    seq_mask = tf.tile(seq_mask, [6, 1, 1])
    sess.run(seq_mask)
    future_mask = future_mask_gen(features, params)
    future_mask = sess.run(future_mask)