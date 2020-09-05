from utils import add_layer_summary, pr_summary_hook
from layers import avg_pooling_embedding_v2
import tensorflow as tf


def model_fn(features, labels, mode, params):
    tokens = tf.reshape( features['tokens'], shape=[ -1, tf.shape(features['tokens'])[1]] )
    extra_features = tf.reshape( features['extra_features'], shape = [-1, params['extra_size']] )

    add_layer_summary('label', labels)

    with tf.variable_scope( 'initialization' ):
        embedding = tf.get_variable(shape = [params['vocab_size'], params['emb_size']],
                                    initializer = tf.truncated_normal_initializer(), name = 'embedding')
        if params['use_extra']:
            w = tf.get_variable( shape = [params['emb_size'] + params['extra_hidden_size'], params['label_size']],
                                initializer=tf.truncated_normal_initializer(), name ='hidden_weight' )
        else:
            w = tf.get_variable( shape = [params['emb_size'], params['label_size']],
                                initializer=tf.truncated_normal_initializer(), name ='hidden_weight' )
        b = tf.get_variable( shape = [params['label_size']],
                            initializer = tf.truncated_normal_initializer(), name = 'hidden_bias')

        for item in [embedding, w, b]:
            add_layer_summary(item.name, item)

    with tf.variable_scope('hidden_layer'):
        dense = avg_pooling_embedding_v2( embedding, tokens, params)  # batch_size * emb_size
        add_layer_summary(dense.name, dense)

        if params['use_extra']:
            extra = tf.layers.dense(extra_features, units = params['extra_hidden_size'], activation = 'relu') # batch_size * extra_hidden_size
            extra = tf.layers.batch_normalization(extra, center = True, scale = True, trainable = True,
                                                  training = (mode == tf.estimator.ModeKeys.TRAIN))
            dense = tf.concat([dense, extra], axis=1 ) # batch_size * (emb_size + extra_hidden_size)
            add_layer_summary( 'extra_features', extra_features )

        logits = tf.matmul(dense, w ) + b  # batch_size * label_size

        add_layer_summary('logits', logits)
        add_layer_summary('sigmoid', tf.sigmoid(logits))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'prediction_prob' : tf.sigmoid(logits)
        }

        return tf.estimator.EstimatorSpec(mode = tf.estimator.ModeKeys.PREDICT,
                                          predictions = predictions)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= labels, logits = logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
        if params['decay_steps'] >0 :
            learning_rate = tf.train.exponential_decay(learning_rate = params['learning_rate'],
                                                       global_step = tf.train.get_global_step(),
                                                       decay_steps = params['decay_steps'],
                                                       decay_rate = params['decay_rate']
                                                       )
        else:
            learning_rate = params['learning_rate']

        optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(cross_entropy, global_step = tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss = cross_entropy, train_op = train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # The eval metric of quora competition is F1 score
        prediction = tf.to_float(tf.greater_equal(tf.sigmoid(logits), 0.33))
        precision, precision_op= tf.metrics.precision(labels, predictions = prediction)
        recall, recall_op= tf.metrics.recall(labels, predictions = prediction) # why op is needed in evaluation
        f1 = 2*(precision * recall)/(precision+recall)

        # add precision-recall curve summary
        summary_hook = pr_summary_hook(logits, labels, num_threshold = 20, output_dir = params['model_dir'], save_steps= 1000 )

        eval_metrics_ops = {
            'accuracy': tf.metrics.accuracy(labels = labels, predictions = prediction),
            'auc': tf.metrics.auc(labels = labels, predictions = tf.sigmoid(logits), curve = 'ROC'),
            'pr': tf.metrics.auc(labels = labels, predictions = tf.sigmoid(logits), curve = 'PR'),
            'precision': (precision, precision_op),
            'recall': (recall, recall_op),
            'f1_score': (f1, tf.identity( f1 ))
        }

        return tf.estimator.EstimatorSpec(mode, loss = cross_entropy,
                                          eval_metric_ops = eval_metrics_ops, evaluation_hooks = [summary_hook])