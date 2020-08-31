import shutil
import tensorflow as tf
import os

from config.default_config import RUN_CONFIG
from tensorboard import summary

def clear_model(model_dir):
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))
    else:
        print( '{} model cleaned'.format(model_dir) )


def build_estimator(params, model_dir, model_fn, gpu_enable=0):
    session_config = tf.ConfigProto()

    if gpu_enable:
        # control CPU and Mem usage
        session_config.gpu_options.allow_growth = True
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        session_config.log_device_placement = True
        session_config.allow_soft_placement = True
        session_config.inter_op_parallelism_threads = 6
        session_config.intra_op_parallelism_threads = 6
        mirrored_strategy = tf.distribute.MirroredStrategy()
    else:
        mirrored_strategy = None

    run_config = tf.estimator.RunConfig(
        save_summary_steps=RUN_CONFIG['save_steps'],
        log_step_count_steps=RUN_CONFIG['log_steps'],
        keep_checkpoint_max=RUN_CONFIG['keep_checkpoint_max'],
        save_checkpoints_steps=RUN_CONFIG['save_steps'],
        session_config=session_config,
        train_distribute=mirrored_strategy, eval_distribute=None
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params,
        model_dir=model_dir
    )

    return estimator

def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag.replace(':','_')), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag.replace(':','_')),  value)

def write_projector_meta(log_dir, dictionary):
    # Projector meta file: word_index \t word
    with open( os.path.join( log_dir, 'met  adata.tsv' ), "w" ) as f:
        f.write( "Index\tLabel\n" )
        for word_index, word in enumerate(dictionary.keys()):
            f.write("%d\t%s\n" % (word_index, word))

def pr_summary_hook(logits, labels, num_threshold, output_dir, save_steps):
    # add precision-recall curve summary
    pr_summary = summary.pr_curve( name='pr_curve',
                                   predictions=tf.sigmoid( logits ),
                                   labels=tf.cast( labels, tf.bool ),
                                   num_thresholds= num_threshold )

    summary_hook = tf.train.SummarySaverHook(
        save_steps= save_steps,
        output_dir= output_dir,
        summary_op=[pr_summary]
    )

    return summary_hook