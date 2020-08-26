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

    input_embedding = tf.stack(input_embedding, name = 'input_embedding_vector') # batch * emb_size
    return input_embedding
