# -*- coding=utf-8 -*-

import tensorflow as tf


def base_inference(features, labels):
    result = {}
    token_table = tf.get_collection('token_table')[0]
    result['input_tokenid']=tf.identity(features['tokens'])
    result['input_token']= tf.identity(token_table.lookup(features['tokens']))
    if labels is not None:
        result['output_tokenid'] = tf.identity(labels['tokens'])
        result['output_token'] = tf.identity(token_table.lookup(labels['tokens']))

    return result


def avg_encode_inference(encoder_output, decoder_output, features, labels):
    """
    vectorization used in universal sentence encoder
    Element-wise sum of the representations at each word position, scaled by (input_length)**0.5
    output:
        batch_size * emb_size
    """
    with tf.variable_scope('avg_encode_inference'):
        vector = tf.reduce_mean(encoder_output.output, axis=1)
        seq_len = tf.expand_dims(features['seq_len'], axis=1)
        vector = tf.divide(vector, seq_len**0.5)

        result = base_inference(features, labels)
        result['vector'] = vector
    return result


def last_encode_inference(encoder_output, decoder_output, features, labels):
    """
    vectorization used in skip_thought
    use last hidden state in encoder as representation
    """
    with tf.variable_scope('last_encode_inference'):
        vector = encoder_output.state
        result = base_inference(features, labels)
        result['vector'] = vector
    return result


def last_encode_decode_inference(encoder_output, decoder_output, features, labels):
    """
    vectorization used in quick thought
    concat last hidden state in encoder and decoder as representation
    """
    vector = tf.concat(encoder_output.state, decoder_output.state, axis=-1)
    result = base_inference(features, labels)
    result['vector'] = vector
    return result


def sequence_inference(encoder_output, decoder_output, features, labels):
    """
    Do NMT/QA style inference, predict each token in decoder output
    output:
        batch_size * decoder_length
    """
    with tf.variable_scope('sequence_inference', reuse=tf.AUTO_REUSE):
        token_table = tf.get_collection('token_table')[0]
        predict_id = tf.argmax(decoder_output.output, axis=-1, output_type=tf.int32)  # batch_size * seq_len
        predict_vocab = token_table.lookup(predict_id)

        result = base_inference(features, labels)
        result['predict_vocab'] = predict_vocab
        result['predict_id'] = predict_id
        return result
