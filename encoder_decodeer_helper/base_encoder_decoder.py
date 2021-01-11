# -*- coding=utf-8 -*-

import tensorflow as tf
from utils import add_layer_summary
from train_utils import get_learning_rate, gradient_clipping


class BaseEncoderDecoder(object):
    def __init__(self, params, encoder_func, decoder_func, loss_func, infer_func):
        self.params = params
        self.encoder_func = encoder_func
        self.decoder_func = decoder_func
        self.loss_func = loss_func
        self.infer_func = infer_func
        self.embedding = None

    def embedding_gen(self, input_):
        with tf.variable_scope('embedding_gen', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(dtype=self.params['dtype'],
                                             initializer=tf.constant(self.params['pretrain_embedding']),
                                             name='word_embedding' )
            tf.add_to_collection('word_embedding', self.embedding)
            add_layer_summary(self.embedding.name, self.embedding)

            seq_emb_input = tf.nn.embedding_lookup(self.embedding, input_['tokens']) # batch_size * max_len * emb_size

        return seq_emb_input

    def encode(self, features, mode):
        with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
            encoder_input = self.embedding_gen(features)
            encoder_output = self.encoder_func(encoder_input, features, self.params, mode)

        return encoder_output

    def decode(self, encoder_output, features, labels, mode):
        with tf.variable_scope('decoding', reuse=tf.AUTO_REUSE):
            decoder_input = self.embedding_gen(labels)
            decoder_output = self.decoder_func(decoder_input, encoder_output, features, labels, self.params, mode)

        return decoder_output

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
            ## TODO teacher forcing evaluation, evaluate what ?
            # EVAL funuc same as prediction only with additional metrics
            predictions = self.infer(encoder_output, decoder_output, features, labels)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        loss = self.compute_loss(encoder_output, decoder_output, labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=get_learning_rate(self.params))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                if self.params['clip_gradient']:
                    train_op = gradient_clipping(optimizer, loss,
                                                 self.params['lower_gradient'], self.params['upper_gradient'])
                else:
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def compute_loss(self, encoder_output, decoder_output, labels):
        loss = self.loss_func(encoder_output, decoder_output, labels, self.params)

        return loss

    def infer(self, encoder_output, decoder_output, features, labels):
        output = self.infer_func(encoder_output, decoder_output, features, labels)

        return output

