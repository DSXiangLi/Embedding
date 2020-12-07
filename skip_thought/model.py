# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
from itertools import chain
from skip_thought.seq2seq_utils import (ENCODER_FAMILY, SEQ_LOSS_OUTPUT,
                                        sequence_loss, agg_sequence_loss, sequence_mask)
from skip_thought.decoder import decoder
from utils import add_layer_summary, build_model_fn_from_class
from train_utils import gradient_clipping, get_learning_rate


class EncoderBase(object):
    def __init__(self, params):
        self.params = params
        self.init()

    def init(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(dtype = self.params['dtype'],
                                             initializer=tf.constant(self.params['pretrain_embedding']),
                                             name='word_embedding' )

            add_layer_summary(self.embedding.name, self.embedding)

    def general_encoder(self, features):
        """
        Apply encoding func for input sequence
        Input
            features:{tokens:, seq_len:}
        Return
            ENCODER_OUTPUT
        """

        encoder = ENCODER_FAMILY[self.params['encoder_type']]

        seq_emb_input = tf.nn.embedding_lookup(self.embedding, features['tokens']) # batch_size * max_len * emb_size

        encoder_output = encoder(seq_emb_input, features['seq_len'], self.params) # batch_size

        return encoder_output

    def vectorize(self, state_list, features):
        with tf.variable_scope('inference'):
            result={}
            # copy through input for checking
            result['input_tokenid']=tf.identity(features['tokens'], name='input_id')
            token_table = tf.get_collection('token_table')[0]
            result['input_token']= tf.identity(token_table.lookup(features['tokens']), name='input_token')

            result['encoder_state'] = tf.concat(state_list, axis = 1, name ='sentence_vector')

        return result


class SkipThought(EncoderBase):
    def __init__(self, params):
        super(SkipThought, self).__init__(params)

    def build_model(self, features, labels, mode):
        """
        Build model_fn for Quick Thought
        Input
            features: {tokens:, seq_len:}
            labels: {tokens:, seq_len:}
        Return
            tf.estimator.EstimatorSpec
        """

        encoder_output = self.encode(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.vectorize([encoder_output.state[0]], features)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        decoder_output = self.decode(encoder_output, labels, mode )

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

    def encode(self, features):
        with tf.variable_scope('encoding'):
            encoder_output = self.general_encoder(features)

            add_layer_summary('encoder_output.state', encoder_output.state)
            add_layer_summary('encoder_output.output', encoder_output.output)

        return encoder_output

    def decode(self, encoder_output, labels, mode):
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


class QuickThought(EncoderBase):
    def __init__(self, params):
        super(QuickThought, self).__init__(params)

    def build_model(self, features, labels, mode):
        """
        Build model_fn for Quick Thought
        Input
            features: {tokens:, seq_len:}
            labels: {tokens:, seq_len:}
        Return
            tf.estimator.EstimatorSpec
        """

        input_encode = self.input_encode(features)

        output_encode = self.output_encode(features, labels, mode)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.vectorize([input_encode.state[0], output_encode.state[0]], features)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions)

        sim_score = tf.matmul(input_encode.state[0], output_encode.state[0], transpose_b=True) # [batch, batch] sim score
        add_layer_summary('sim_score', sim_score)

        loss = self.compute_loss(sim_score)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=get_learning_rate(self.params))

            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                if self.params['clip_gradient']:
                    train_op = gradient_clipping(optimizer, loss,
                                                 self.params['lower_gradient'], self.params['upper_gradient'])
                else:
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec( mode, loss=loss, train_op=train_op )

    def input_encode(self, features):
        with tf.variable_scope('input_encoding', reuse=False):
            encoder_output = self.general_encoder(features)

            add_layer_summary('state', encoder_output.state)
            add_layer_summary('output', encoder_output.output)
        return encoder_output

    def output_encode(self, features, labels, mode):
        """
        For quick thought, decode will be another encoder with different parameters and do inner-product with encoder
        Return
            [batch, batch] inner product of encoder_state * decoder_state
        """
        with tf.variable_scope('output_encoding', reuse=False):
            if mode == tf.estimator.ModeKeys.PREDICT:
                encoder_output = self.general_encoder(features)
            else:
                encoder_output=self.general_encoder(labels)

            add_layer_summary('state', encoder_output.state)
            add_layer_summary('output', encoder_output.output)
        return encoder_output

    def compute_loss(self, sim_score):
        """
        compute log perplexity loss per batch, mask padded value
        Input:
            decoder_output: [batch, batch] inner product
            labels: not needed. 0/1 label is pre-defined in the diagonal matrix, with diag being 1 and all other being 0
        """

        with tf.variable_scope('compute_loss'):
            batch_size = sim_score.get_shape().as_list()[0]
            sim_score = tf.matrix_set_diag(sim_score, np.zeros(batch_size))

            # create targets: set element within diagonal offset to 1
            targets = np.zeros(shape = (batch_size, batch_size))
            offset = self.params['context_size']//2 ## offset of the diagonal
            for i in chain(range(1, 1+offset), range(-offset, -offset+1)):
                diag = np.diagonal(targets, offset = i)
                diag.setflags(write=True)
                diag.fill(1)

            targets = targets/np.sum(targets, axis=1, keepdims = True)

            targets = tf.constant(targets, dtype = self.params['dtype'])

            losses = tf.nn.softmax_cross_entropy_with_logits(labels = targets,
                                                             logits = sim_score)

            losses = tf.reduce_mean(losses)

        return losses


skip_thought_model = build_model_fn_from_class(SkipThought)
quick_thought_model = build_model_fn_from_class(QuickThought)
