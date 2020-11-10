# -*- coding=utf-8 -*-
from collections import namedtuple
import tensorflow as tf

SEQ_LOSS_OUTPUT = namedtuple('SEQ_LOSS_OUTPUT', ['loss_id', 'loss_scaler', 'loss_per_time', 'loss_per_batch'])
SEQ_PRED_OUTPUT = namedtuple('SEQ_PRED_OUTPUT', ['predict_prob', 'predict_id', 'predict_tokens', 'seq_len', 'vector'])
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


def sequence_mask(decoder_output, labels, params, mode):
    """
    Create Padded mask for sequence.
    In train, mask padded part in batch padded length.
    In eval, mask padded part in batch decoding length
    Input:
        labels, params, mode
    Output:
        mask: Train: (batch * padded_len-1), Eval: (batch * decode_len)
    """

    if mode == tf.estimator.ModeKeys.TRAIN:
        max_len = tf.shape( labels['tokens'] )[1]
        # length -1: because decoder source exclude end_token, target exclude start_token
        mask = tf.sequence_mask( lengths=tf.to_int32( labels['seq_len'] - 1 ), maxlen=max_len - 1,
                                 dtype=params['dtype'], name='loss_seq_mask' )
    else:
        max_len = tf.reduce_max(decoder_output.seq_len)
        # No minus 1 is needed, because for predict=1 dataset only do prepend
        mask = tf.sequence_mask( lengths=tf.to_int32( labels['seq_len'] ), maxlen=max_len,
                                 dtype=params['dtype'], name='loss_seq_mask' )
    return mask


def sequence_loss(logits, target, mask, mode):
    """
    Weighted cross-entropy loss for a sequence of logits
    Input:
        logits: batch_size * decode_len/(padded_len-1) * n_class
        target: batch_size * padded_len
        mask: mask for padded value, batch_size * decode_len/(padded_len-1)
    Output:
        loss_mat: (batch * decode_len/(padded_len-1)) * 1
    """
    with tf.variable_scope('Sequence_loss_matrix'):
        n_class = tf.shape(logits)[2]
        decode_len = tf.shape(logits)[1] # used for infer only, max_len is determined by decoder
        logits = tf.reshape(logits, [-1, n_class])

        if mode == tf.estimator.ModeKeys.TRAIN:
            # In train, target
            target = tf.reshape(target[:, 1:], [-1]) # (batch * (padded_len-1)) * 1
        elif mode == tf.estimator.ModeKeys.EVAL:
            # In eval, target has paded_len, logits have decode_len
            target = tf.reshape(target[:, : decode_len], [-1]) # batch * (decode_len) *1
        else:
            raise Exception('sequence loss is only used in train or eval, not in pure prediction')
        loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target, logits = logits)
        loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1])) # apply padded mask on output loss

    return loss_mat


def agg_sequence_loss(loss_mat, mask,  axis):
    """
    Aggregate sequence loss into scaler, along different axis, using differnt op
    Input:
        loss_max: (batch_size * max_len) * 1
        mask : batch_size * max_len of real length for each sequence
        axis: batch/time/scaler
    Return:
        loss: axis=batch/time loss has rank1 ,axis=scaler output scaler
    """
    with tf.variable_scope('Loss_{}'.format(axis)):
        if axis == 'scaler':
            loss = tf.reduce_sum(loss_mat)
            n_sample = tf.reduce_sum(mask)
            loss = loss/n_sample
        else:
            loss_mat = tf.reshape(loss_mat, tf.shape(mask)) # (batch_size * max_len) * 1-> batch_size * max_len

            if axis == 'batch':
                loss = tf.reduce_sum(loss_mat, axis=1) # batch
                n_sample = tf.reduce_sum(mask, axis=1) # batch
                loss = tf.math.divide_no_nan(loss, n_sample) # batch
            elif axis == 'time':
                loss = tf.reduce_sum(loss_mat, axis=0) # max_len
                n_sample = tf.reduce_sum(mask, axis=0) # max_len
                loss = tf.math.divide_no_nan(loss, n_sample) # max_len
            else:
                raise Exception('Only scaler/batch/time are supported in axis param')

    return loss

