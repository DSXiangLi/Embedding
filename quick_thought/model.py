# -*-coding:utf-8 -*-
import tensorflow as tf

from utils import add_layer_summary, build_model_fn_from_class
from encoder_decodeer_helper.base_encoder_decoder import BaseEncoderDecoder
from encoder_decodeer_helper.loss import neighbour_cls_loss
from encoder_decodeer_helper.inference import last_encode_decode_inference
from encoder_decodeer_helper.encoder import rnn_encoder


class QuickThought(BaseEncoderDecoder):
    def __init__(self, params,  loss_func, infer_func):
        super(QuickThought, self).__init__(params,  loss_func, infer_func)

    def encode(self, features, mode):
        """
        RNN Encoder
        """
        with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
            encoder_input = self.embedding_func(features['tokens'])

            encoder_output = rnn_encoder(encoder_input, features['seq_len'], self.params)

            add_layer_summary('state', encoder_output.state)
            add_layer_summary('output', encoder_output.output)

        return encoder_output

    def decode(self, encoder_output, features, labels, mode):
        """
        inner product decoder
        """
        with tf.variable_scope('decoding', reuse=tf.AUTO_REUSE):
            decoder_output = self.encode(labels, mode)

        return decoder_output


model_fn = build_model_fn_from_class(QuickThought,
                                     loss_func=neighbour_cls_loss,
                                     infer_func=last_encode_decode_inference)