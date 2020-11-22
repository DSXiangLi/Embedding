# -*- coding=utf-8 -*-
import tensorflow as tf

from skip_thought.seq2seq_utils import (ENCODER_FAMILY, DECODER_FAMILY, SEQ_LOSS_OUTPUT,
                                        sequence_loss, agg_sequence_loss, sequence_mask)
from utils import add_layer_summary, build_model_fn_from_class
from train_utils import gradient_clipping, get_learning_rate


class QuickThought(object):
    def __init__(self, params):
        self.params = params
        self.init()

    def init(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(dtype = self.params['dtype'],
                                             initializer=tf.constant(self.params['pretrain_embedding']),
                                             name='word_embedding' )

            add_layer_summary(self.embedding.name, self.embedding)

    def build_model(self, features, labels, mode):
        """
        Build model_fn for Quick Thought
        Input
            features: {tokens:, seq_len:}
            labels: {tokens:, seq_len:}
        Return
            tf.estimator.EstimatorSpec
        """

        encoder_output = self._encode(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.predict(encoder_output, features)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        decoder_output = self._decode(encoder_output, labels, mode )

        loss_output = self.compute_loss( decoder_output, labels, mode )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=get_learning_rate(self.params))

            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                if self.params['clip_gradient']:
                    train_op = gradient_clipping(optimizer, loss_output.loss_scaler,
                                                 self.params['lower_gradient'], self.params['upper_gradient'])
                else:
                    train_op = optimizer.minimize(loss_output.loss_scaler, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec( mode, loss=loss_output.loss_scaler, train_op=train_op )

        if mode == tf.estimator.ModeKeys.EVAL:
            ##TODO: add configurable training and evaluation hook
            return tf.estimator.EstimatorSpec(mode, loss=loss_output.loss_scaler)

    def _encode(self, features):
        """
        Apply encoding func for input sequence
        Input
            features:{tokens:, seq_len:}
        Return
            ENCODER_OUTPUT
        """
        with tf.variable_scope('encoding'):
            encoder = ENCODER_FAMILY[self.params['encoder_type']]

            seq_emb_input = tf.nn.embedding_lookup(self.embedding, features['tokens']) # batch_size * max_len * emb_size

            encoder_output = encoder(seq_emb_input, features['seq_len'], self.params) # batch_size

            add_layer_summary('encoder_output.state', encoder_output.state )
            add_layer_summary( 'encoder_output.output', encoder_output.output )
        return encoder_output

    def _decode(self, encoder_output, labels, mode):
        """
        Apply decoding func for target sequence. If train, use train decoder, else use infer decoder.
        Input
            encoder_output: ENCODER_OUTPUT
            features: {tokens:, seq_len:}
            labels: {tokens:, seq_len:}
            mode: tf.estimator.ModeKeys
        Return
            DECODER_OUTPUT
        """
        with tf.variable_scope('decoding'):
            decoder = DECODER_FAMILY[self.params['decoder_type']]

            if mode == tf.estimator.ModeKeys.TRAIN:
                seq_emb_output = tf.nn.embedding_lookup(self.embedding, labels['tokens']) # batch_size * max_len * emb_size
                input_len = labels['seq_len']
            elif mode == tf.estimator.ModeKeys.EVAL:
                seq_emb_output = None
                input_len = labels['seq_len']
            else:
                seq_emb_output = None
                input_len = None

            decoder_output = decoder(encoder_output, seq_emb_output, input_len,\
                                     self.embedding, self.params, mode)

            add_layer_summary('decoder_output.state', decoder_output.state )
            add_layer_summary('decoder_output.output', decoder_output.output.rnn_output)

        return decoder_output

    def compute_loss(self, decoder_output, labels, mode):
        """
        compute log perplexity loss per batch, mask padded value
        Input:
            decoder_output: DECODER_OUTPUT
            labels: {tokens:, seq_len:}
        """
        with tf.variable_scope('compute_loss'):
            mask = sequence_mask(decoder_output, labels, self.params, mode)

            loss_mat = sequence_loss(logits=decoder_output.output.rnn_output,
                                     target=labels['tokens'],
                                     mask=mask,
                                     mode=mode)
            loss = []
            for axis in ['scaler', 'batch', 'time']:
                loss.append(agg_sequence_loss(loss_mat, mask, axis))

        return SEQ_LOSS_OUTPUT(loss_id=loss_mat,
                               loss_scaler=loss[0],
                               loss_per_batch=loss[1],
                               loss_per_time=loss[2])

    def predict(self, encoder_output, features):
        """
        Generate prediction given decoder_output

        """
        with tf.variable_scope('inference'):
            vector = tf.identity( encoder_output.state[0], name='sentence_vector' )
            input_id = tf.identity(features['tokens'], name='input_id')

            token_table = tf.get_collection('token_table')[0]
            input_token = tf.identity(token_table.lookup(features['tokens']), name='input_token')

        return {
                'encoder_state': vector,
                'input_tokenid': input_id,
                'input_token': input_token
                }


model_fn = build_model_fn_from_class(QuickThought)
