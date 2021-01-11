# -*- coding=utf-8 -*-

import tensorflow as tf

from utils import add_layer_summary, build_model_fn_from_class
from transformer.tools import positional_encoding, future_mask_gen, multi_head_attention, ffn
from encoder_decodeer_helper.base_encoder_decoder import BaseEncoderDecoder
from encoder_decodeer_helper.tools import DECODER_OUTPUT, ENCODER_OUTPUT, token2sequence, seq_mask_gen
from encoder_decodeer_helper.loss import sequence_loss
from encoder_decodeer_helper.inference import sequence_inference


class Transformer(BaseEncoderDecoder):
    def __init__(self, params, encoder_func, decoder_func, loss_func, infer_func):
        super(Transformer, self).__init__(params, encoder_func, decoder_func, loss_func, infer_func)

    def gen_embedding(self, input_, mode):
        """
        transform input token into embedding and add positional encoding to it
        input
            input_: batch * seq_len
        output
            batch * seq_len * emb_dim
        """
        # token embedding lookup & scale by emb_size
        seq_emb_input = super().embedding_gen(input_) # batch * seq_len * emb_dim

        d_model = tf.shape(seq_emb_input)[-1]
        seq_emb_input = seq_emb_input / d_model**0.5

        # add positional encoding and dropout in training
        pos_encoding = positional_encoding(seq_emb_input) # seq_len * emb_dim

        seq_emb = tf.layers.dropout(tf.add(seq_emb_input, pos_encoding),
                                    rate=self.params['dropout_rate'],
                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

        return seq_emb


def transformer_encoder(encoder_input, features, params, mode):
    """
    6 idential layer consisting of multiheaad attention + add&norm + feed forward + add&norm
    input
        encoder_input: batch * seq_len * emb_dim
        features: dict {'tokens':, 'seq_len':}
    output
        encoder_output: dimension unchanged after transformation
    """
    self_mask = seq_mask_gen(features, params)

    encoder_output=None
    for i in range(params['encode_attention_layers']):
        with tf.variable_scope('attention_layer_{}'.format(i), reuse=True):
            encoder_output = multi_head_attention(key=encoder_input, query=encoder_input, value=encoder_input,
                                                  mask=self_mask, params=params, mode=mode)
            add_layer_summary('self_attention_output', encoder_output)

            encoder_output = ffn(encoder_output, params, mode)
            add_layer_summary('ffn', encoder_output)

    return ENCODER_OUTPUT(output=encoder_output, state=encoder_output[:, -1, :])


def transformer_decoder_block(decoder_input, encoder_output, features, labels, params, mode):
    """
    6 idential layer consisting of multiheaad attention + add&norm + feed forward + add&norm
    input
        encoder_input: batch * seq_len * emb_dim
        features: dict {'tokens':, 'seq_len':}
    output
        encoder_output: dimension unchanged after transformation
    """
    self_mask = future_mask_gen(labels,  params)
    encoder_mask = seq_mask_gen(features, params)

    decoder_output=None
    for i in range(params['decode_attention_layers']):
        with tf.variable_scope('attention_layer_{}'.format(i), reuse=True):
            decoder_output = multi_head_attention(key=decoder_input, value=decoder_input,
                                                  query=decoder_input, mask=self_mask,
                                                  params=params, mode=mode)
            add_layer_summary('self_attention_output', decoder_output)

            decoder_output = multi_head_attention(key=encoder_output.output, value=encoder_output.output,
                                                  query=decoder_output, mask=encoder_mask,
                                                  params=params, mode=mode)
            add_layer_summary('encoder_attention_output', decoder_output)

            decoder_output = ffn(decoder_output, params, mode)
            add_layer_summary('ffn', decoder_output)

    # use share embedding weight for linear project from emb_size to vocab_size
    embedding = tf.get_collection('word_embedding')[0]
    # batch_size * seq_len * emb_size -> batch_size * seq_len * vocab_size
    logits = tf.matmul(decoder_output, embedding, transpose_b=True)

    return DECODER_OUTPUT(output=logits, state=decoder_output,
                          seq_len=labels['seq_len'])


def transformer_decoder(decoder_input, encoder_output, features, labels, params, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        # teacher forcing like self-attention is used in training, remove <end_token> in decode source
        labels['seq_len'] = labels['seq_len']
        labels['tokens'] = labels['tokens']
        decoder_output = transformer_decoder_block(decoder_input, encoder_output, features, labels, params, mode)
    else:
        # when real label exists, visualize on tensorboard
        if mode == tf.estimator.ModeKeys.EVAL:
            tf.summary.text('decode_target', token2sequence(labels['tokens'][0, :]))

        # initialize iter with start token
        decoder_output = None
        labels['tokens'] = labels['tokens'][:, 0] # batch_size * 1 <start_token>
        labels['seq_len'] = tf.ones_like(labels['tokens']) # batch_size * 1 <start_token>
        decoder_input = decoder_input[:, 0, :] # batch_size * 1 * emb_size

        # iteratively add decode prediction to decode source
        embedding = tf.get_collection('word_embedding')[0]
        for i in range(params['max_decode_iter']):
            decoder_output = transformer_decoder_block(decoder_input, encoder_output, features, labels, params, mode)

            predict_vocab = tf.argmax(decoder_output.output[:, -1, :], axis=-1, output_type=tf.int32) # batch_size * 1

            # if all predict vocab in batch = end_token stop iter
            if tf.reduce_all(tf.equal(predict_vocab, params['end_token'])):
                break
            # add predict token to decode input
            decoder_input = tf.concat([decoder_input,
                                       embedding.lookup(predict_vocab)], axis=2) # batch_size * (i+1) * emb_size
            labels['tokens'] = tf.concat([labels['tokens'], predict_vocab], axis=1) # batch_size * (i+1)
            labels['seq_len'] = labels['seq_len'] + tf.where(tf.equal(predict_vocab, params['end_token']),
                                                             tf.zeros_like(predict_vocab),
                                                             tf.ones_like(predict_vocab)) # add for non-ending sequence

        if mode == tf.estimator.ModeKeys.EVAL:
            tf.summary.text('decode_prediction', token2sequence(labels['tokens'][0, :]))

    return decoder_output


model_fn = build_model_fn_from_class(Transformer,
                                     encoder_func=transformer_encoder,
                                     decoder_func=transformer_decoder,
                                     loss_func=sequence_loss,
                                     infer_func=sequence_inference)
