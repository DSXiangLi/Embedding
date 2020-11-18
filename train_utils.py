# -*- coding=utf-8 -*-
import tensorflow as tf


def gradient_clipping(optimizer, cost, _lower, _upper):
    """
    apply gradient clipping
    """
    gradients, variables = zip(*optimizer.compute_gradients( cost ))

    clip_grad = [tf.clip_by_value( grad, _lower, _upper ) for grad in gradients if grad is not None]

    train_op = optimizer.apply_gradients(zip(clip_grad, variables),
                                         global_step=tf.train.get_global_step() )

    return train_op


def get_learning_rate(params):
    """
    exponential learning rate deccay
    """
    ## TODO: support other decay method

    if params.get('rate_decay', False):
        return tf.train.exponential_decay(params['learning_rate'],
                                          tf.train.get_global_step(),
                                          params['decay_rate'],
                                          params['decay_step']
                                      )
    else:
        return params['learning_rate']
