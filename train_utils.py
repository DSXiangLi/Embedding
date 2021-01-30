# -*- coding=utf-8 -*-
import tensorflow as tf
from utils import add_layer_summary


def gradient_clipping(optimizer, cost, _lower, _upper):
    """
    apply gradient clipping
    """
    gradients, variables = zip(*optimizer.compute_gradients( cost ))

    clip_grad = [tf.clip_by_value( grad, _lower, _upper ) for grad in gradients if grad is not None]

    train_op = optimizer.apply_gradients(zip(clip_grad, variables),
                                         global_step=tf.train.get_global_step() )

    return train_op


def get_train_op(optimizer, loss, params):
    if params.get('clip_gradient', False):
        train_op = gradient_clipping(optimizer, loss,
                                     params['lower_gradient'],
                                     params['upper_gradient'])
    else:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op


def get_learning_rate(params):
    """
    all sorts of learning rate strategy
    """
    ## TODO: support other decay method

    if params.get('rate_decay', False):
        lr = exponential_decay(params)
    elif params.get('warmup', False):
        lr = noam(params)
    else:
        lr = params['learning_rate']
    tf.summary.scalar('learning_rate', lr)
    return lr


def exponential_decay(params):
    PARAMS = ['learning_rate', 'decay_rate', 'decay_step']
    assert all([ i in params.keys() for i in PARAMS]), '{} are needed fro exponential decay'.format(','.join(PARAMS))

    lr = tf.train.exponential_decay(params['learning_rate'],
                                    tf.train.get_global_step(),
                                    params['decay_rate'],
                                    params['decay_step']
                                    )
    return lr


def noam(params):
    PARAMS = ['emb_size', 'warmup_steps']
    assert all([ i in params.keys() for i in PARAMS]), '{} are needed for noam'.format(','.join(PARAMS))
    lr = params['emb_size'] ** -0.5 * tf.minimum(tf.cast(tf.train.get_global_step(), params['dtype']) ** -0.5,
                                                tf.cast(tf.train.get_global_step(), params['dtype']) * params[
                                                    'warmup_steps'] ** -1.5)

    return lr
