# -*- coding=utf-8 -*-
from collections import namedtuple
import tensorflow as tf

SEQ_LOSS_OUTPUT = namedtuple('SEQ_LOSS_OUTPUT', ['loss_id', 'loss_scaler', 'loss_per_time', 'loss_per_batch'])
SEQ_PRED_OUTPUT = namedtuple('SEQ_PRED_OUTPUT', ['predict_prob', 'predict_id', 'predict_tokens', 'seq_len', 'vector'])
DECODER_OUTPUT = namedtuple('DecoderOutput', ['output', 'state', 'seq_len'])
ENCODER_OUTPUT = namedtuple('EncoderOutput', ['output', 'state'])


ENCODER_FAMILY = {}


def encoder_decoder_collection(func_name):
    def helper(func):
        if 'encoder' in func_name:
            ENCODER_FAMILY[func_name] = func
        else:
            raise Exception('Only encoder are supported in func_name')
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
        cells = [ tf.nn.rnn_cell.DropoutWrapper(cell = cell_class(num_units = params['hidden_units'][i]),
                                                output_keep_prob=params['keep_prob'][i],
                                                state_keep_prob= params['keep_prob'][i]) for i in range(params['cell_size']) ]
    )


def embedding_func(embedding):
    def helper(id):
        return tf.nn.embedding_lookup(embedding, id)
    return helper


def future_mask_gen(key_dict, params):
    """
    In decoder self-attention, additional future_mask is needed to avoid leakage. add future mask on padding mask
    input:
        inbput_dict: {'tokens':, 'seq_len'}, seq_len is non-padded sequence len
    output:
        mask:  batch_size * input_len * input_len, with 1 to keep, 0 to drop
    """
    # give 0 to all padding position for both key and query
    seq_mask = seq_mask_gen(key_dict, params)
    # batch_size * key_len * key_len(seq_len)
    mask = tf.matmul(tf.expand_dims(seq_mask, axis=2), tf.expand_dims(seq_mask, axis=1))
    # keep upper triangle with diagonal
    mask = tf.matrix_band_part(mask, numm_lower=0, num_upper=-1)

    return mask


def seq_mask_gen(key_dict, params):
    """
    Default paddinng mask for sequence
    input:
        input_dict: {'tokens':, 'seq_len'}, seq_len is non-padded sequence len
    output:
        mask: batch * 1 * key_len, with 1 to keep,0 to drop
    """
    mask = tf.sequence_mask(length=tf.to_int32(key_dict['seq_len']), maxlen=tf.shape(key_dict['tokens'])[1],
                            dtype=params['dtype'])

    # add axis1 to enable broadcast to query_len * key_len later in scaled-dot
    mask = tf.expand_dims(mask, axis=1)

    return mask


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


def token2sequence(tokens):
    token_table = tf.get_collection('token_table')[0]
    words = token_table.lookup(tokens)
    return tf.constant(' '.join(words))

