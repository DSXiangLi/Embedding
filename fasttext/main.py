import sys
print(sys.path)
import argparse
import importlib
import pickle
import tensorflow as tf
from utils import clear_model, build_estimator
from fasttext.dataset import FasttextDataset
from config.default_config import CHECKPOINT_DIR


def main(args):

    model_dir = CHECKPOINT_DIR.format(args.data, args.model)
    data_file = './data/{}/train.tfrecords'.format(args.data)
    dict_file = './data/{}/dictionary.pkl'.format( args.data )

    if args.clear_model:
        clear_model(model_dir)

    # Init config
    TRAIN_PARAMS = getattr(importlib.import_module('config.{}_config'.format(args.data)), 'TRAIN_PARAMS')
    RUN_CONFIG = getattr( importlib.import_module( 'config.{}_config'.format( args.data ) ), 'RUN_CONFIG' )

    # Init dataset
    input_pipe = FasttextDataset(data_file = data_file,
                                 dict_file = dict_file,
                                 epochs = TRAIN_PARAMS['epochs'],
                                 batch_size = TRAIN_PARAMS['batch_size'],
                                 min_count = TRAIN_PARAMS['min_count'],
                                 buffer_size = TRAIN_PARAMS['buffer_size'],
                                 invalid_index = TRAIN_PARAMS['invalid_index'],
                                 padded_shape = TRAIN_PARAMS['padded_shape'],
                                 padding_values = TRAIN_PARAMS['padding_values'],
                                 ngram = TRAIN_PARAMS['ngram']

                                 )
    input_pipe.build_dictionary()

    TRAIN_PARAMS.update(
        {
            'vocab_size': input_pipe.vocab_size,
            'freq_dict': input_pipe.dictionary,
            'model_dir': model_dir
        }
    )

    # Init Estimator
    model_fn = getattr(importlib.import_module('model_{}'.format(args.model)), 'model_fn')
    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn, args.gpu, RUN_CONFIG)

    if args.step == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator,
            metric_name ='loss',
            max_steps_without_decrease= 100 * 1000
        )

        train_spec = tf.estimator.TrainSpec( input_fn = input_pipe.build_dataset(is_predict =0),
                                             hooks = [early_stopping])

        eval_spec = tf.estimator.EvalSpec( input_fn = input_pipe.build_dataset(is_predict=1),
                                           steps = 500,
                                           throttle_secs=60 )

        tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec )

    if args.step == 'predict':
        prediction = estimator.predict( input_fn = input_pipe.build_dataset(is_predict=1))
        with open('prediction.pkl', 'wb') as f:
            pickle.dump(prediction, f)

if __name__ == '__main__':
    tf.logging.set_verbosity( tf.logging.ERROR )

    parser = argparse.ArgumentParser()
    parser.add_argument( '--step', type=str, help='train or predict',
                         required=False, default = 'train' )
    parser.add_argument( '--clear_model', type=int, help= 'Whether to clear existing model',
                         required=False, default=1)
    parser.add_argument( '--data', type=str, help='which data to use[data should be list of tokenized string]',
                         required=False, default='quora')
    parser.add_argument( '--model', type = str, help = 'which model to use[fasttext | textcnn | textrnn]',
                         required=False, default='fasttext')
    parser.add_argument('--gpu', type =int, help = 'Whether to enable gpu',
                        required =False, default = 0 )
    args = parser.parse_args()

    main(args)
