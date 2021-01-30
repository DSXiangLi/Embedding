# -*- coding=utf-8 -*-
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.util import nest

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


def init_input_embedding(embedding):
    def helper(tokens):
        return tf.nn.embedding_lookup(embedding, tokens)
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
    """
    convert list ot token_id to string, for inference and tf.summary
    """
    token_table = tf.get_collection('token_table')[0]
    words = token_table.lookup(tokens)
    token2str = lambda x:  ' '.join([i.decode('utf-8') for i in x ])
    return tf.py_func(token2str, [words], tf.string)


def bridge(encoder_output, decoder_cell):
    """
    Bridge encoder & decoder when state not match, 2 possible mismatch reason
    1. hidden size differ(including cell size for rnn encoder)
    2. cell type differ: gru vs lstm

    Ref from tf-seq2seq: https://github.com/google/seq2seq/blob/master/seq2seq/models/bridges.py
    """
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


