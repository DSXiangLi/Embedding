import shutil
import tensorflow as tf


def clear_model(model_dir):
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))
    else:
        print( '{} model cleaned'.format(model_dir) )


def build_estimator(params, model_dir, model_fn):

    run_config = tf.estimator.RunConfig(
        save_summary_steps=50,
        log_step_count_steps=50,
        keep_checkpoint_max=3,
        save_checkpoints_steps=50
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
        params = params ,
        model_dir = model_dir
    )

    return estimator

def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag.replace(':','_')), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag.replace(':','_')),  value)

