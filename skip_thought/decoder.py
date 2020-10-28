# -*- coding=utf-8 -*-
"""
Decoder Collection

"""
import tensorflow as tf
from collections import namedtuple
import tensorflow.contrib.seq2seq as seq2seq
from skip_thought.utils import encoder_decoder_collection, build_rnn_cell, embedding_func

DECODER_OUTPUT = namedtuple('DecoderOutput', ['output', 'state', 'seq_len'])


def general_train_decoder(decoder_cell, encoder_output, input_emb, input_len, embedding, params):
    train_helper = seq2seq.TrainingHelper( inputs=input_emb[:,:-1], # batch_size * max_len * emb_size
                                           sequence_length=input_len -1,
                                           time_major=False,
                                           name='training_helper' )

    decoder = seq2seq.BasicDecoder( cell=decoder_cell,
                                    helper=train_helper,
                                    initial_state=encoder_output.state,
                                    output_layer=None )  ## TODO when to use output layer
    return decoder


def general_infer_decoder(decoder_cell, encoder_output, input_emb, input_len, embedding, params):
    batch_size = tf.shape(input_len)[0]

    if params['beam_width'] >1 :
        # If beam search multiple prediction are uesd at each time step
        decoder = seq2seq.BeamSearchDecoder( cell=decoder_cell,
                                             embedding=embedding_func( embedding ),
                                             initial_state=encoder_output,
                                             beam_width=params['beam_width'],
                                             start_tokens=tf.fill([batch_size], params['start_token']),
                                             end_token=params['end_token'],
                                             output_layer=None )
    else:
        # otherwise only argmax samplid is used at each time step
        infer_helper = seq2seq.GreedyEmbeddingHelper( embedding=embedding_func( embedding ),
                                                      start_tokens=tf.fill([batch_size], params['start_token']),
                                                      end_tokens=params['end_tokens'] )

        decoder = seq2seq.BasicDecoder( cell=decoder_cell,
                                        helper=infer_helper,
                                        initial_state=encoder_output.state,
                                        output_layer=None )
    return decoder


@encoder_decoder_collection('gru_decoder')
def gru_decoder(encoder_output, input_emb, input_len, embedding, params, mode):
    gru_cell = build_rnn_cell( 'gru', params )

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_func = general_train_decoder
        max_iteration = None
    else:
        decoder_func = general_infer_decoder
        max_iteration = params['max_decoder_iter']  # only infer max_decoder_iter

    decoder = decoder_func(gru_cell, encoder_output, input_emb, input_len, embedding, params)

    output, state, seq_len = seq2seq.dynamic_decode(decoder=decoder,
                                                    output_time_major=True,
                                                    impute_finished=True,
                                                    maximum_iterations=max_iteration) ## TODO 返回是max_len还是seq_len return max_len * batch_size * hidden_size

    return DECODER_OUTPUT(output = output, state = state, seq_len = seq_len)
