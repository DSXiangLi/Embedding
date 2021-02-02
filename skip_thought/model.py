# -*-coding:utf-8 -*-
import tensorflow as tf

from utils import add_layer_summary, build_model_fn_from_class
from encoder_decodeer_helper.base_encoder_decoder import BaseEncoderDecoder
from encoder_decodeer_helper.tools import token2sequence, DECODER_OUTPUT
from encoder_decodeer_helper.loss import sequence_loss
from encoder_decodeer_helper.inference import last_encode_inference
from encoder_decodeer_helper.encoder import rnn_encoder
from encoder_decodeer_helper.decoder import decoder


class SkipThought(BaseEncoderDecoder):
    def __init__(self, params,  loss_func, infer_func):
        super(SkipThought, self).__init__(params,  loss_func, infer_func)

    def encode(self, features, mode):
        """
        RNN Encoder
        """
        with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
            encoder_input = self.embedding_func(features['tokens'])

            encoder_output = rnn_encoder(encoder_input, features['seq_len'], self.params)

            add_layer_summary('encoder_output.state', encoder_output.state)
            add_layer_summary('encoder_output.output', encoder_output.output)

        return encoder_output

    def decode(self, encoder_output, features, labels, mode):
        """
        RNN Decoder
        """
        with tf.variable_scope('decoding', reuse=tf.AUTO_REUSE):

            decoder_output = decoder(encoder_output, labels, self.embedding_func, self.params, mode)

            if mode != tf.estimator.ModeKeys.PREDICT:
                tf.summary.text('decode_source', token2sequence(labels['tokens'][0, :-1]))
                tf.summary.text('decode_prediction', token2sequence(tf.cast(decoder_output.output.sample_id[0, :], tf.int32)))

            return DECODER_OUTPUT(output=decoder_output.output.rnn_output, state=decoder_output.state,
                                  seq_len=decoder_output.seq_len)


model_fn = build_model_fn_from_class(SkipThought,
                                     loss_func=sequence_loss,
                                     infer_func=last_encode_inference)