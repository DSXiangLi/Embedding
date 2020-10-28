import tensorflow as tf

from word2vec.hierarchysoftmax import HierarchySoftmax
from utils import add_layer_summary
from layers import avg_pooling_embedding


def negative_sampling(mode, output_embedding, bias, labels, input_embedding_vector, params):
    """
    supported : sampled_loss, nce_loss
    train mode : binary classification
    eval mode : multi-class classification
    stratify: only sample in the same category
    """

    if mode == tf.estimator.ModeKeys.TRAIN:
        if params['loss'] == 'nce_loss':
            loss = tf.nn.nce_loss(
                weights=output_embedding,  # vocab_size * emb_dim
                biases=bias,  # vocab_size
                labels=labels,  # batch_size * 1
                inputs=input_embedding_vector,  # batch_size * emb_dim
                num_sampled=params['ng_sample'],
                num_classes=params['vocab_size'],
                num_true=1,
                name='nce_loss',
                partition_strategy='div'
            )
        else:
            loss = tf.nn.sampled_softmax_loss(
                    weights = output_embedding, # vocab_size * emb_dim
                    biases = bias, #vocab_size
                    labels = labels, #batch_size * 1
                    inputs = input_embedding_vector, #batch_size * emb_dim
                    num_sampled = params['ng_sample'],
                    num_classes = params['vocab_size'],
                    num_true = 1,
                    seed = 1234,
                    name = 'sampled_loss',
                    partition_strategy ='div' # how the matrix kept in memory
                ) # batch * 1
        loss = tf.reduce_mean(loss)
    else:
        # (batch, emb_dim) * (emb_dim, vocab_size) + (vocab_size)
        logits = tf.matmul( input_embedding_vector, tf.transpose( output_embedding ) ) + bias

        labels_one_hot = tf.one_hot( labels, params['vocab_size'] )

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits )
        )

    return loss


def model_fn(features, labels, mode, params):

    assert params['model'] in ['CBOW','SG'], 'For moedl only [CBOW | SG] are supported'
    assert params['train_algo'] in ['HS','NS'], 'For train_algo only [HS | NS] are supported'

    if params['train_algo'] == 'NS':
        assert params['loss'] in ['nce_loss','sample_loss'], 'For Negative Sampling loss only [nce_loss|sample_loss] are supported'

    if params['train_algo'] == 'HS':
        # If Hierarchy Softmax is used, initialize a huffman tree first
        hstree = HierarchySoftmax( params['freq_dict'], params['pad_index'] )
        hstree.build_tree()
        hstree.traverse()
        hstree.convert2tensor()

    if params['model'] == 'CBOW':
        features = tf.reshape(features, shape = [-1, 2 * params['window_size']])
        labels = tf.reshape(labels, shape = [-1,1])
    else:
        features = tf.reshape(features, shape = [-1,])
        labels = tf.reshape(labels, shape = [-1,1])

    with tf.variable_scope( 'initialization' ):
        w0 = tf.get_variable( shape=[params['vocab_size'], params['emb_size']],
                              initializer=tf.truncated_normal_initializer(), name='input_word_embedding' )
        if params['train_algo'] == 'HS':
            w1 = tf.get_variable( shape=[hstree.num_node, params['emb_size']],
                                  initializer=tf.truncated_normal_initializer(), name='hierarchy_node_embedding_' )
            b1 = tf.get_variable( shape = [hstree.num_node],
                                  initializer=tf.random_uniform_initializer(), name = 'bias')
        else:
            w1 = tf.get_variable( shape=[params['vocab_size'], params['emb_size']],
                                  initializer=tf.truncated_normal_initializer(), name='output_word_embedding' )
            b1 = tf.get_variable( shape=[params['vocab_size']],
                                  initializer=tf.random_uniform_initializer(), name='bias')
        for item in [w0, w1, b1]:
            add_layer_summary(item.name, item)

    with tf.variable_scope('input_hidden'):
        # batch_size * emb_size
        if params['model'] == 'CBOW':
            input_embedding_vector = avg_pooling_embedding(w0, features, params)
        else:
            input_embedding_vector = tf.nn.embedding_lookup(w0, features, name = 'input_embedding_vector')
        add_layer_summary(input_embedding_vector.name, input_embedding_vector)

    with tf.variable_scope('hidden_output'):
        if params['train_algo'] == 'HS':
            loss = hstree.get_loss( input_embedding_vector, labels, w1, b1, params)
        else:
            loss = negative_sampling(mode = mode,
                                     output_embedding = w1,
                                     bias = b1,
                                     labels = labels,
                                     input_embedding_vector =input_embedding_vector,
                                     params = params)

    optimizer = tf.train.AdamOptimizer( learning_rate = params['learning_rate'] )
    update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )

    with tf.control_dependencies( update_ops ):
        train_op = optimizer.minimize( loss, global_step= tf.train.get_global_step() )

    return tf.estimator.EstimatorSpec( mode, loss=loss, train_op=train_op )


