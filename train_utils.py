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
