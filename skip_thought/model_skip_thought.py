# -*- coding=utf-8 -*-
import tensorflow as tf

from skip_thought.seq2seq_utils import (ENCODER_FAMILY, DECODER_FAMILY, SEQ_LOSS_OUTPUT, SEQ_PRED_OUTPUT,
                                        sequence_loss, agg_sequence_loss, sequence_mask)
from skip_thought.BaseModel import Seq2SeqModel

from utils import add_layer_summary, build_model_fn_from_class


class QuickThought(Seq2SeqModel):
    def __init__(self, params):
        super(QuickThought, self).__init__(params)

    def params_check(self):
        pass

    def init(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(shape=[self.params['vocab_size'], self.params['emb_size']], dtype = self.params['dtype'],
                                             initializer=tf.truncated_normal_initializer(), name='word_embedding' )

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
        decoder_output = self._decode(encoder_output, labels, mode )

        predictions = self.predict( decoder_output)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        loss_output = self.compute_loss( decoder_output, labels, mode )

        if mode == tf.estimator.ModeKeys.TRAIN:
            ##TODO: gradient cliping. optimizer.apply_gradiennts(capped_gradient)
            ##TODO: extract get_learning_rate to utils
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])

            train_op = optimizer.minimize(loss_output.loss_scaler, global_step=tf.train.get_global_step() )

            return tf.estimator.EstimatorSpec( mode, loss=loss_output.loss_scaler, train_op=train_op )

        if mode == tf.estimator.ModeKeys.EVAL:
            ##TODO: add configurable training and evaluation hook
            #eval_metric_ops = self.get_eval_metric(decoder_output, labels)

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
            else:
                seq_emb_output = None

            decoder_output = decoder(encoder_output, seq_emb_output, labels['seq_len'],\
                                     self.embedding, self.params, mode)

            add_layer_summary('decoder_output.state', decoder_output.state )

        return decoder_output

    def compute_loss(self, decoder_output, labels, mode):
        """
        compute loss per batch, mask padded value
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

    def predict(self, decoder_output):
        """
        Generate prediction given decoder_output
        """
        with tf.variable_scope('inference'):
            predict_prob = tf.nn.softmax(decoder_output.output.rnn_output) # batch_size * decoder_length * vocab_size
            predict_id = decoder_output.output.sample_id # batch_size * decoder_length
            predict_tokens = tf.get_collection('token_table')[0].lookup(predict_id) # batch * decoder_length

        return SEQ_PRED_OUTPUT(predict_prob=predict_prob,
                               predict_id=predict_id,
                               predict_tokens=predict_tokens,
                               seq_len=decoder_output.seq_len)

    def eval_metric(self, decoder_output):
        pass


model_fn = build_model_fn_from_class(QuickThought)
