import shutil
import tensorflow as tf
import os
import pickle

from tensorboard import summary
from tensorflow.python.client import device_lib


def clear_model(model_dir):
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))
    else:
        print( '{} model cleaned'.format(model_dir) )


def build_estimator(params, model_dir, model_fn, gpu_enable, RUN_CONFIG):
    session_config = tf.ConfigProto()

    if gpu_enable:
        # control CPU and Mem usage
        session_config.gpu_options.allow_growth = RUN_CONFIG['allow_growth']
        session_config.gpu_options.per_process_gpu_memory_fraction = RUN_CONFIG['pre_process_gpu_fraction']
        session_config.log_device_placement = RUN_CONFIG['log_device_placement']
        session_config.allow_soft_placement = RUN_CONFIG['allow_soft_placement']
        session_config.inter_op_parallelism_threads = RUN_CONFIG['inter_op_parallel']
        session_config.intra_op_parallelism_threads = RUN_CONFIG['intra_op_parallel']
        mirrored_strategy = tf.distribute.MirroredStrategy(RUN_CONFIG['devices'])
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

    if os.path.isdir(model_dir):
        warm_start_dir = model_dir
    else:
        warm_start_dir = None

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params,
        model_dir=model_dir,
        warm_start_from=warm_start_dir
    )

    return estimator


def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag.replace(':','_')), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag.replace(':','_')),  value)


def write_projector_meta(log_dir, dict_dir):
    with open(os.path.join(dict_dir, 'dictionary.pkl'), 'rb') as f:
        dictionary = pickle.load(f)

    # Projector meta file: word_index \t word
    with open( os.path.join( log_dir, 'metadata.tsv' ), "w" ) as f:
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


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def build_model_fn_from_class(model_class, **kwargs):
    def model_fn(features, labels, mode, params):
        # params is passed during run time
        model_cls = model_class(params, **kwargs)
        return model_cls.build_model(features, labels, mode)
    return model_fn
