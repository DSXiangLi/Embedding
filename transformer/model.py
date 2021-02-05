# -*- coding=utf-8 -*-

import tensorflow as tf

from utils import add_layer_summary, build_model_fn_from_class
from transformer.tools import positional_encoding, future_mask_gen, multi_head_attention, ffn, init_input_embedding
from encoder_decodeer_helper.base_encoder_decoder import BaseEncoderDecoder
from encoder_decodeer_helper.tools import token2sequence, seq_mask_gen
from encoder_decodeer_helper.loss import sequence_loss
from encoder_decodeer_helper.inference import sequence_inference
from encoder_decodeer_helper.encoder import ENCODER_OUTPUT
from encoder_decodeer_helper.decoder import DECODER_OUTPUT


class Transformer(BaseEncoderDecoder):
    def __init__(self, params,  loss_func, infer_func):
        self.pos_encoding = None
        super(Transformer, self).__init__(params,  loss_func, infer_func)
        self.embedding_func = init_input_embedding(self.embedding, self.pos_encoding, self.params) # apply word embedding+pos encocding

    def init_embedding(self):
        # token embedding lookup & scale by emb_size
        super().init_embedding()  # batch * seq_len * emb_dim

        # scale embedding by d_model^0.5
        d_model = tf.cast(tf.shape(self.embedding)[-1], self.params['dtype'])
        self.embedding = self.embedding * (d_model**0.5)
        self.pos_encoding = positional_encoding(d_model=self.params['emb_size'], max_len=self.params['max_len'],
                                                dtype=self.params['dtype'])

    def encode(self, features, mode):
        """
        6 idential layer consisting of multiheaad attention + add&norm + feed forward + add&norm
        input
            features: dict {'tokens':, 'seq_len':}
        output
            encoder_output: dimension unchanged after transformation
        """
        with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
            encoder_input = self.embedding_func(features['tokens'], mode) # batch * seq_len * emb_size
            self_mask = seq_mask_gen(features, self.params)

            for i in range(self.params['encode_attention_layers']):
                with tf.variable_scope('attention_layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                    encoder_input = multi_head_attention(key=encoder_input, query=encoder_input, value=encoder_input,
                                                         mask=self_mask, params=self.params, mode=mode)
                    add_layer_summary('self_attention_output', encoder_input)

                    encoder_input = ffn(encoder_input, self.params, mode)
                    add_layer_summary('ffn', encoder_input)

        return ENCODER_OUTPUT(output=encoder_input, state=encoder_input[:, -1, :])

    def _decode_helper(self, encoder_output, features, labels, mode):
        """
        6 attention layer consisting of self and encoder attention layer
        input
            encoder_input: batch * seq_len * emb_dim
            features: dict {'tokens':, 'seq_len':}
        output
            encoder_output: dimension unchanged after transformation
        """
        decoder_input = self.embedding_func(labels['tokens'], mode)  # batch * seq_len * emb

        self_mask = future_mask_gen(labels, self.params)
        encoder_mask = seq_mask_gen(features, self.params)

        for i in range(self.params['decode_attention_layers']):
            with tf.variable_scope('attention_layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
                    decoder_input = multi_head_attention(key=decoder_input, value=decoder_input,
                                                         query=decoder_input, mask=self_mask,
                                                         params=self.params, mode=mode)
                    add_layer_summary('self_attention_output', decoder_input)
                with tf.variable_scope('encode_attention', reuse=tf.AUTO_REUSE):
                    decoder_input = multi_head_attention(key=encoder_output.output, value=encoder_output.output,
                                                         query=decoder_input, mask=encoder_mask,
                                                         params=self.params, mode=mode)
                    add_layer_summary('encoder_attention_output', decoder_input)

                decoder_input = ffn(decoder_input, self.params, mode)
                add_layer_summary('ffn', decoder_input)

        # use share embedding weight for linear project from emb_size to vocab_size
        logits = tf.matmul(decoder_input, self.embedding, transpose_b=True)  # seq_len * emb_size->seq_len * vocab_size

        return DECODER_OUTPUT(output=logits, state=decoder_input, seq_len=labels['seq_len'])

    def decode(self, encoder_output, features, labels, mode):
        """
        Call Decoder Helper to do decode in train or predict
        Train: teacher forcing is used
        Predict: Iter through input sequence
        """
        with tf.variable_scope('decoding', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                # teacher forcing is used in training
                decoder_output = self._decode_helper(encoder_output, features, labels, mode)
            else:
                # initialize iter with start token
                labels = {}
                decoder_output = None
                batch_size = tf.shape(features['tokens'])[0]
                labels['tokens'] = tf.cast((tf.ones((batch_size, 1)) * self.params['start_index']),
                                           tf.int32)  # batch_size * 1 : <start_token>
                labels['seq_len'] = tf.ones(batch_size, dtype=self.params['dtype'])  # batch_size
                # iteratively add decode prediction to decode source

                for i in range(self.params['max_decode_iter']):
                    decoder_output = self._decode_helper(encoder_output, features, labels, mode)
                    # add predict token to decode source
                    predict_vocab = tf.argmax(decoder_output.output[:, -1, :], axis=-1,
                                              output_type=tf.int32)  # (batch_size, )
                    labels['tokens'] = tf.concat([labels['tokens'], tf.expand_dims(predict_vocab, axis=1)], axis=1)  # (batch_size, i+1)
                    labels['seq_len'] = labels['seq_len'] + tf.where(tf.equal(predict_vocab, self.params['end_index']),
                                                                     tf.zeros(batch_size, dtype=self.params['dtype']),
                                                                     tf.ones(batch_size, dtype=self.params['dtype']))  # (batch_size,)

            if mode != tf.estimator.ModeKeys.PREDICT:
                tf.summary.text('decode_source', token2sequence(labels['tokens'][0, :-1]))
                tf.summary.text('decode_prediction', token2sequence(tf.argmax(decoder_output.output[0, :, :],
                                                                              axis=-1, output_type=tf.int32)))

        return decoder_output


model_fn = build_model_fn_from_class(Transformer,
                                     loss_func=sequence_loss,
                                     infer_func=sequence_inference)
