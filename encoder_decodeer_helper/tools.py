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



def seq_mask_gen(input_, params):
    """
    Default paddinng mask for sequence
    input:
        input_: {'tokens':, 'seq_len'}, seq_len is non-padded sequence len
    output:
        mask: batch * 1 * key_len, with 1 to keep,0 to drop
    """
    mask = tf.sequence_mask(lengths=tf.to_int32(input_['seq_len']), maxlen=tf.shape(input_['tokens'])[1],
                            dtype=params['dtype'])

    # add axis1 to enable broadcast to query_len * key_len later in scaled-dot
    mask = tf.expand_dims(mask, axis=1)

    return mask


def token2sequence(tokens):
    token_table = tf.get_collection('token_table')[0]
    words = token_table.lookup(tokens)
    return tf.constant(' '.join(words))

