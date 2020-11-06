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


def general_train_decoder(decoder_cell, encoder_output, input_emb, input_len, embedding, output_layer, params):
    #In training remove end_token
    train_helper = seq2seq.TrainingHelper( inputs=input_emb, # batch_size * max_len-1 * emb_size
                                           sequence_length=input_len-1, # exclude last token
                                           time_major=False,
                                           name='training_helper' )

    decoder = seq2seq.BasicDecoder( cell=decoder_cell,
                                    helper=train_helper,
                                    initial_state=encoder_output.state,
                                    output_layer=output_layer )
    return decoder


def general_infer_decoder(decoder_cell, encoder_output, input_emb, input_len, embedding, output_layer, params):
    batch_size = tf.shape(input_len)[0]

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
        # otherwise only argmax samplid is used at each time step
        infer_helper = seq2seq.GreedyEmbeddingHelper( embedding=embedding_func( embedding ),
                                                      start_tokens=tf.fill([batch_size], params['start_token']),
                                                      end_token=params['end_token'] )

        decoder = seq2seq.BasicDecoder( cell=decoder_cell,
                                        helper=infer_helper,
                                        initial_state=encoder_output.state,
                                        output_layer=output_layer )
    return decoder


@encoder_decoder_collection('gru_decoder')
def gru_decoder(encoder_output, input_emb, input_len, embedding, params, mode):
    gru_cell = build_rnn_cell( 'gru', params )

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_func = general_train_decoder
        max_iteration = None
    elif mode == tf.estimator.ModeKeys.EVAL:
        decoder_func = general_infer_decoder
        max_iteration = tf.reduce_max(input_len) # decode max sequence length(=padded_length)in EVAL
    else:
        decoder_func = general_infer_decoder
        max_iteration = params['max_decoder_iter']  # decode pre-defined max_decode iter in predict

    output_layer = tf.layers.Dense( params['vocab_size'] )  # used for helper sample

    decoder = decoder_func(gru_cell, encoder_output, input_emb, input_len, embedding, output_layer, params)

    output, state, seq_len = seq2seq.dynamic_decode(decoder=decoder,
                                                    output_time_major=False,
                                                    impute_finished=True,
                                                    maximum_iterations=max_iteration)

    return DECODER_OUTPUT(output = output, state = state, seq_len = seq_len)
