# -*- coding=utf-8 -*-
"""
Decoder Collection
    1. gru decoder
    2. lstm decoder
    3. bi-lstm decoder
    4. all above with attention wrapepr
"""
import tensorflow as tf
from collections import namedtuple
import tensorflow.contrib.seq2seq as seq2seq

from skip_thought.seq2seq_utils import encoder_decoder_collection, build_rnn_cell, embedding_func

DECODER_OUTPUT = namedtuple('DecoderOutput', ['output', 'state', 'seq_len'])


def get_helper(encoder_output, input_emb, input_len, batch_size, embedding, mode, params):

    if mode == tf.estimator.ModeKeys.TRAIN:
        if params['conditional']:
            # conditional train helper with encoder output state as direct input
            # Reshape encoder state as auxiliary input: 1* batch_size * hidden -> batch_size * max_len * hidden
            decoder_length = tf.shape(input_emb)[1]
            state_shape = tf.shape(encoder_output.state)
            encoder_state = tf.tile(tf.reshape(encoder_output.state, [state_shape[1],
                                                                      state_shape[0],
                                                                      state_shape[2]]),
                                    [1, decoder_length, 1])
            # stupid way: use auxiliary input from ScheduleOutput Helper
            # next_inputs_fn = tf.layers.Dense(units=params['emb_size'], trainable =False)
            # sample output has same shape as embedding, not used
            #
            # helper = seq2seq.ScheduledOutputTrainingHelper(inputs=input_emb,  # batch_size * max_len-1 * emb_size
            #                                                sequence_length=input_len-1,  # exclude last token
            #                                                time_major=False,
            #                                                # must be float: 0-> sampling 100% teacher forcing
            #                                                sampling_probability=tf.Variable(0.0, dtype=tf.float32),
            #                                                seed=1234,
            #                                                auxiliary_inputs=encoder_state,
            #                                                next_inputs_fn=next_inputs_fn,
            #                                                name='training_helper')
            # smarter way: concat encoder_state with input_emb directly
            input_emb = tf.concat([encoder_state, input_emb], axis=-1)

            helper = seq2seq.TrainingHelper( inputs=input_emb, # batch_size * max_len-1 * emb_size
                                             sequence_length=input_len-1, # exclude last token
                                             time_major=False,
                                             name='training_helper' )
    else:
        helper = seq2seq.GreedyEmbeddingHelper( embedding=embedding_func( embedding ),
                                                start_tokens=tf.fill([batch_size], params['start_token']),
                                                end_token=params['end_token'] )

    return helper


def get_decoder(decoder_cell, encoder_output, input_emb, input_len, embedding, output_layer, mode, params):
    batch_size = tf.shape(encoder_output.output)[0]
    if params['beam_width'] >1 :
        # If beam search multiple prediction are uesd at each time step
        decoder = seq2seq.BeamSearchDecoder( cell=decoder_cell,
                                             embedding=embedding_func( embedding ),
                                             initial_state=encoder_output,
                                             beam_width=params['beam_width'],
                                             start_tokens=tf.fill([batch_size], params['start_token']),
                                             end_token=params['end_token'],
                                             output_layer=output_layer )

    else:
        helper = get_helper(encoder_output, input_emb, input_len, batch_size, embedding, mode, params)

        decoder = seq2seq.BasicDecoder( cell=decoder_cell,
                                        helper=helper,
                                        initial_state=encoder_output.state,
                                        output_layer=output_layer )

    return decoder


@encoder_decoder_collection('gru_decoder')
def gru_decoder(encoder_output, input_emb, input_len, embedding, params, mode):
    gru_cell = build_rnn_cell( 'gru', params )

    if mode == tf.estimator.ModeKeys.TRAIN:
        max_iteration = None
    elif mode == tf.estimator.ModeKeys.EVAL:
        max_iteration = tf.reduce_max(input_len) # decode max sequence length(=padded_length)in EVAL
    else:
        max_iteration = params['max_decode_iter']  # decode pre-defined max_decode iter in predict

    output_layer=tf.layers.Dense(units=params['vocab_size'])  # used for infer helper sample or train loss calculation
    decoder = get_decoder(gru_cell, encoder_output, input_emb, input_len, embedding, output_layer, mode, params)

    output, state, seq_len = seq2seq.dynamic_decode(decoder=decoder,
                                                    output_time_major=False,
                                                    impute_finished=True,
                                                    maximum_iterations=max_iteration)

    return DECODER_OUTPUT(output=output, state = state, seq_len=seq_len)

