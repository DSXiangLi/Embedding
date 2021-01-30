# -*- coding=utf-8 -*-
from typing import Dict
import tensorflow as tf

from utils import add_layer_summary
from train_utils import get_learning_rate, get_train_op
from encoder_decodeer_helper.tools import ENCODER_OUTPUT, DECODER_OUTPUT, init_input_embedding


class BaseEncoderDecoder(object):
    def __init__(self, params, loss_func, infer_func):
        self.params = params
        self.loss_func = loss_func
        self.infer_func = infer_func
        self.embedding = None
        self.init_embedding()
        # wrap up embedding func, default to embedding lookup without mode
        self.embedding_func = init_input_embedding(self.embedding)

    def init_embedding(self):
        with tf.variable_scope('embedding_gen', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(dtype=self.params['dtype'],
                                             initializer=tf.constant(self.params['pretrain_embedding']),
                                             name='word_embedding' )
            add_layer_summary(self.embedding.name, self.embedding)

    def encode(self, features, mode) -> ENCODER_OUTPUT:
        raise NotImplementedError()

    def decode(self, encoder_output, features, labels, mode) -> DECODER_OUTPUT:
        raise NotImplementedError()

    def build_model(self, features, labels, mode):
        encoder_output = self.encode(features, mode)

        # For vectorization type task, decoder is not needed in inference
        if (mode != tf.estimator.ModeKeys.TRAIN) & (self.params['skip_decoder']):
            decoder_output = None
        else:
            decoder_output = self.decode(encoder_output, features, labels, mode)

        if mode in tf.estimator.ModeKeys.PREDICT:
            predictions = self.infer(encoder_output, decoder_output, features, labels)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        if mode in tf.estimator.ModeKeys.EVAL:
            predictions = self.infer(encoder_output, decoder_output, features, labels)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        loss = self.compute_loss(encoder_output, decoder_output, labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=get_learning_rate(self.params))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = get_train_op(optimizer, loss, self.params)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def compute_loss(self, encoder_output, decoder_output, labels) -> float:
        # same model graph can apply to different loss function
        loss = self.loss_func(encoder_output, decoder_output, labels, self.params)

        return loss

    def infer(self, encoder_output, decoder_output, features, labels) -> Dict:
        # same model graph can apply to different infer function
        output = self.infer_func(encoder_output, decoder_output, features, labels)

        return output

