import shutil
import tensorflow as tf
import os

from config.default_config import RUN_CONFIG

def clear_model(model_dir):
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))
    else:
        print( '{} model cleaned'.format(model_dir) )


def build_estimator(params, model_dir, model_fn):

    run_config = tf.estimator.RunConfig(
        save_summary_steps = RUN_CONFIG['save_steps'],
        log_step_count_steps = RUN_CONFIG['log_steps'],
        keep_checkpoint_max = RUN_CONFIG['keep_checkpoint_max'],
        save_checkpoints_steps = RUN_CONFIG['save_steps']
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

def write_projector_meta(log_dir, dictionary):
    # Projector meta file: word_index \t word
    with open( os.path.join( log_dir, 'metadata.tsv' ), "w" ) as f:
        f.write( "Index\tLabel\n" )
        for word_index, word in enumerate(dictionary.keys()):
            f.write("%d\t%s\n" % (word_index, word))

