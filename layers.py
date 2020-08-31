import tensorflow as tf
from config.default_config import *

def avg_pooling_embedding(embedding, features, params):
    """
    :param features: (batch, 2*window_size)
    :param embedding: (vocab_size, emb_size)
    :return: input_embedding : average pooling of context embedding
    """
    input_embedding= []
    samples = tf.unstack(features, params['batch_size'])
    for sample in samples:
        sample = tf.boolean_mask(sample, tf.not_equal(sample, INVALID_INDEX), axis=0) # (real_size,)
        tmp = tf.nn.embedding_lookup(embedding, sample) # (real_size, emb_size)
        input_embedding.append(tf.reduce_mean(tmp, axis=0)) # (emb_size, )

    input_embedding = tf.stack(input_embedding, name = 'input_central_embeddinng') # batch * emb_size
    return input_embedding


def avg_pooling_embedding_v2(embedding, features):
    """
    Allow Embedding for INVALID Index and apply weighting mask
    :param features: (batch, 2*window_size)
    :param embedding: (vocab_size, emb_size)
    :return: input_embedding : average pooling of context embedding
    """
    input_embedding = tf.nn.embedding_lookup(embedding, features) # batch * padded_size * emb_size

    zero_mask = tf.expand_dims(tf.equal(features, INVALID_INDEX), axis=2) # batch * padded_size * 1

    weight = tf.where(zero_mask, tf.zeros_like(zero_mask, dtype=tf.float32), tf.ones_like(zero_mask, dtype = tf.float32)) # batch * padded_size *1

    input_embedding = tf.reduce_mean(tf.multiply(weight, input_embedding), axis=1 ) # batch * emb_size

    return input_embedding


