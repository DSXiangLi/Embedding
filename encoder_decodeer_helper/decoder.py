"""
Decoder Collection
1. conditonal gru decoder
2. lstm decoder
"""

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from encoder_decodeer_helper.tools import build_rnn_cell, bridge, DECODER_OUTPUT


def get_helper(encoder_output, input_emb, input_len, embedding_func, mode, params):
    batch_size = tf.shape(encoder_output.output)[0]

    if mode == tf.estimator.ModeKeys.TRAIN:
        if params['conditional']:
            # conditional train helper with encoder output state as direct input
            # Reshape encoder state as auxiliary input: batch_size * hidden -> batch_size * max_len * hidden
            decoder_length = tf.shape(input_emb)[1]
            state_shape = tf.shape(encoder_output.state)
            encoder_state = tf.tile(tf.reshape(encoder_output.state, [state_shape[1],
                                                                      state_shape[0],
                                                                      state_shape[2]]),
                                    [1, decoder_length, 1])

            input_emb = tf.concat([encoder_state, input_emb], axis=-1)

        helper = seq2seq.TrainingHelper( inputs=input_emb, # batch_size * max_len * emb_size
                                         sequence_length=input_len, # exclude last token
                                         time_major=False,
                                         name='training_helper' )
    else:
        helper = seq2seq.GreedyEmbeddingHelper( embedding=embedding_func,
                                                start_tokens=tf.fill([batch_size], params['start_index']),
                                                end_token=params['end_index'] )

    return helper


def decoder(encoder_output, labels, embedding_func, params, mode):
    decoder_cell = build_rnn_cell( params['decoder_cell'], params=params['decoder_cell_params'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        seq_emb_input = embedding_func(labels['tokens'])  # batch_size * max_len * emb_size
        input_len = labels['seq_len']
        max_iteration = None
    else:
        max_iteration = params['max_len'] # decode pre-defined max_decode iter in predict
        seq_emb_input = None
        input_len = None

    output_layer = tf.layers.Dense(units=params['vocab_size'])

    helper = get_helper(encoder_output, seq_emb_input, input_len, embedding_func, mode, params)

    # IF encoder_cell==decoder_cell no bridge needed
    if params.get('bridge_needed', False):
        initial_state = bridge(encoder_output, decoder_cell)
    else:
        initial_state = encoder_output.state

    decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                   helper=helper,
                                   initial_state=initial_state,
                                   output_layer=output_layer)

    output, state, seq_len = seq2seq.dynamic_decode(decoder=decoder,
                                                    output_time_major=False,
                                                    impute_finished=True,
                                                    maximum_iterations=max_iteration)

    return DECODER_OUTPUT(output=output, state=state, seq_len=seq_len)
