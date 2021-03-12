# -*-coding:utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def model2embedding(model, layer_name):
    """
    Extract Embedding layer from model by layer name
    """
    base_model = model.get_layer(layer_name)

    input = layers.Input(shape=(), dtype=tf.string, name='input_source')

    x = base_model(input)

    model = Model(input, x)

    return model


class NBatchLogger(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display=10):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in logs.keys():
            self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()
