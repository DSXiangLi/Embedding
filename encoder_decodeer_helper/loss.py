# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
from itertools import chain

from encoder_decodeer_helper.tools import seq_mask_gen
from utils import add_layer_summary


def agg_sequence_loss(loss_mat, mask,  axis):
    """
    Aggregate sequence loss into scaler, along different axis, using differnt op
    Input:
        loss_max: (batch_size * max_len) * 1
        mask : batch_size * max_len of real length for each sequence
        axis: batch/time/scaler
    Return:
        loss: axis=batch/time loss has rank1 ,axis=scaler output scaler
    """
    with tf.variable_scope('Loss_{}'.format(axis)):
        if axis == 'scaler':
            loss = tf.reduce_sum(loss_mat)
            n_sample = tf.reduce_sum(mask)
            loss = loss/n_sample
        else:
            loss_mat = tf.reshape(loss_mat, tf.shape(mask)) # (batch_size * max_len) * 1-> batch_size * max_len

            if axis == 'batch':
                loss = tf.reduce_sum(loss_mat, axis=1) # batch
                n_sample = tf.reduce_sum(mask, axis=1) # batch
                loss = tf.math.divide_no_nan(loss, n_sample) # batch
            elif axis == 'time':
                loss = tf.reduce_sum(loss_mat, axis=0) # max_len
                n_sample = tf.reduce_sum(mask, axis=0) # max_len
                loss = tf.math.divide_no_nan(loss, n_sample) # max_len
            else:
                raise Exception('Only scaler/batch/time are supported in axis param')

    return loss


def sequence_loss(encoder_output, decoder_output, labels, params):
    """
    seq2seq loss, eval teacher forcing predict on each token
    Label: remove first token, which is <go>
    predict: remove last token, which is beyond <eos>
    """
    with tf.variable_scope('sequence_loss'):
        # batch_size * decode_len * vocab_size -> (batch_size * decode_len) * vocab_size
        n_class = tf.shape(decoder_output.output)[2]
        logits = tf.reshape(decoder_output.output[:, :-1, :], [-1, n_class])

        # in train mode, target is decoder target with <start> token removed
        labels['tokens'] = labels['tokens'][:, 1:]
        labels['seq_len'] = labels['seq_len'] - 1
        # flatten: (batch_size * (pad_len -1)) * 1
        target = tf.reshape(labels['tokens'], [-1])
        mask = seq_mask_gen(labels, params)
        loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
        loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))  # apply padded mask on output loss

    return agg_sequence_loss(loss_mat, mask, axis='scaler')


def neighbour_cls_loss(encoder_output, decoder_output, labels, params):
    """
    Quick thought like loss function: source is continuous sentence, target are the same as input.
    positive samples are the pair within widonw_size around diagonal, all the other sample in batch are negative sample
    """
    sim_score = tf.matmul(encoder_output.state[0], decoder_output.state[0],
                          transpose_b=True)  # [batch, batch] sim score
    add_layer_summary(sim_score.name, sim_score)

    with tf.variable_scope('neighbour_similarity_loss'):
        batch_size = sim_score.get_shape().as_list()[0]
        sim_score = tf.matrix_set_diag(sim_score, np.zeros(batch_size))# ignore self-similarity

        # create targets: set element within diagonal offset to 1
        targets = np.zeros(shape=(batch_size, batch_size))
        offset = params['window_size']  ## offset of the diagonal
        for i in chain(range(1, 1+offset), range(-offset, -offset+1)):
            diag = np.diagonal(targets, offset=i)
            diag.setflags(write=True)
            diag.fill(1)

        targets = targets/np.sum(targets, axis=1, keepdims=True) # normalize target probability to 1

        targets = tf.constant(targets, dtype=params['dtype'])

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                         logits=sim_score)

        losses = tf.reduce_mean(losses)

    return losses
