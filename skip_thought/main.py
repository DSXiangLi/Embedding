import argparse
import importlib
import pickle
import tensorflow as tf
from utils import clear_model, build_estimator
from skip_thought.dataset import SkipThoughtDataset
from config.default_config import CHECKPOINT_DIR, DICTIONARY_DIR
from skip_thought.model import model_fn


def main(args):

    model_dir = CHECKPOINT_DIR.format(args.data, args.model)
    dict_file = DICTIONARY_DIR.format( args.data )
    data_file = {
        'encoder' : './data/{}/encoder_source.txt'.format(args.data),
        'decoder' : './data/{}/decoder_source.txt'.format(args.data)
    }

    if args.clear_model:
        clear_model(model_dir)

    # Init config
    TRAIN_PARAMS = getattr(importlib.import_module('config.{}_config'.format(args.data)), 'TRAIN_PARAMS')
    RUN_CONFIG = getattr( importlib.import_module( 'config.{}_config'.format( args.data ) ), 'RUN_CONFIG' )
    MySpecialToken = getattr( importlib.import_module( 'config.{}_config'.format( args.data ) ), 'MySpecialToken')

    # Init dataset
    input_pipe = SkipThoughtDataset( data_file = data_file,
                                     dict_file = dict_file,
                                     epochs = TRAIN_PARAMS['epochs'],
                                     batch_size = TRAIN_PARAMS['batch_size'],
                                     min_count = TRAIN_PARAMS['min_count'],
                                     max_count = TRAIN_PARAMS['max_count'],
                                     buffer_size = TRAIN_PARAMS['buffer_size'],
                                     special_token = MySpecialToken,
                                     max_len = TRAIN_PARAMS['max_decode_iter']
                                     )
    input_pipe.build_dictionary()

    TRAIN_PARAMS.update(
        {
            'vocab_size': input_pipe.total_size,
            'freq_dict': input_pipe.dictionary,
            'pad_index': input_pipe.pad_index,
            'model_dir': model_dir,
            'start_token': input_pipe.start_token,
            'end_token': input_pipe.end_token,
            'pretrain_embedding': input_pipe.load_pretrain_embedding()
        }
    )

    # Init Estimator
    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn, args.gpu, RUN_CONFIG)

    if args.step == 'train':
        # early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
        #     estimator,
        #     metric_name ='loss',
        #     max_steps_without_decrease= 10 * 20000
        # )

        estimator.train(input_fn = input_pipe.build_dataset())

        # train_spec = tf.estimator.TrainSpec( input_fn = input_pipe.build_dataset(),
        #                                      hooks = [early_stopping])
        #
        # eval_spec = tf.estimator.EvalSpec( input_fn = input_pipe.build_dataset(is_predict=1),
        #                                    steps = 5000,
        #                                    throttle_secs=60)
        #
        # tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec )

    if args.step == 'predict':
        prediction = estimator.predict( input_fn = input_pipe.build_dataset(is_predict=1))
        predict_token = []
        sentence_embedding = []
        for item in prediction:
            predict_token.append(item['predict_tokens'])
            sentence_embedding.append('encoder_state')

        with open( './data/{}/predict_tokens.txt'.format( args.data), 'w', encoding='utf-8' ) as f:
            for line in predict_token:
                f.write( ' '.join( [i.decode('utf-8') for i in line] ).lower() )
                f.write( '\n' )

        with open('./data/{}/predict_embedding.pkl'.format(args.data), 'wb') as f:
            pickle.dump(sentence_embedding, f)


if __name__ == '__main__':
    tf.logging.set_verbosity( tf.logging.ERROR )

    parser = argparse.ArgumentParser()
    parser.add_argument( '--step', type=str, help='train or predict',
                         required=False, default = 'train' )
    parser.add_argument( '--clear_model', type=int, help= 'Whether to clear existing model',
                         required=False, default=1)
    parser.add_argument( '--data', type=str, help='which data to use[data should be list of tokenized string]',
                         required=False, default='bookcorpus')
    parser.add_argument( '--model', type = str, help = 'which model to use[skip_thought | quick_thought]',
                         required=False, default='skip_thought')
    parser.add_argument('--gpu', type =int, help = 'Whether to enable gpu',
                        required =False, default = 0 )
    args = parser.parse_args()

    main(args)


class args:
    data = 'bookcorpus'
    model = 'skip_thought'
    gpu = 0
    clear_model = 0
