
import argparse
import importlib
import pickle
import tensorflow as tf
from utils import clear_model, build_estimator
from encoder_decodeer_helper.dataset import Seq2SeqDataset
from config.default_config import CHECKPOINT_DIR, DICTIONARY_DIR
from transformer.model import model_fn


def main(args):
    model_dir = CHECKPOINT_DIR.format(args.data, args.model)
    dict_file = DICTIONARY_DIR.format(args.data)

    if args.step == 'train':
        data_file = {
            'encoder': './data/{}/train_encoder_source.txt'.format(args.data),
            'decoder': './data/{}/train_decoder_source.txt'.format(args.data)
        }
    else:
        data_file = {
            'encoder': './data/{}/dev_encoder_source.txt'.format(args.data),
            'decoder': './data/{}/dev_decoder_source.txt'.format(args.data)
        }

    if args.clear_model:
        clear_model(model_dir)

    # Init config
    TRAIN_PARAMS = getattr(importlib.import_module('config.{}_config'.format(args.data)), 'TRAIN_PARAMS')
    RUN_CONFIG = getattr(importlib.import_module('config.{}_config'.format(args.data)), 'RUN_CONFIG')
    MySpecialToken = getattr(importlib.import_module('config.{}_config'.format(args.data)), 'MySpecialToken')

    # Init dataset
    input_pipe = Seq2SeqDataset(data_file=data_file,
                                dict_file=dict_file,
                                epochs=TRAIN_PARAMS['epochs'],
                                batch_size=TRAIN_PARAMS['batch_size'],
                                min_count=TRAIN_PARAMS['min_count'],
                                max_count=TRAIN_PARAMS['max_count'],
                                buffer_size=TRAIN_PARAMS['buffer_size'],
                                special_token=MySpecialToken,
                                max_len=TRAIN_PARAMS['max_decode_iter'],
                                min_len=TRAIN_PARAMS['min_len'],
                                pretrain_model=TRAIN_PARAMS['pretrain_model']
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

    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn, args.gpu, RUN_CONFIG)

    if args.step == 'train':
        estimator.train(input_fn=input_pipe.build_dataset())

    if args.step == 'predict':
        # Please disable GPU in prediction to avoid DST exhausted Error
        prediction = estimator.predict(input_fn=input_pipe.build_dataset(is_predict=1))
        res = {}
        for item in prediction:
            res[' '.join([i.decode('utf-8') for i in item['input_token']])] = item['encoder_state']

        with open('./data/{}/{}_predict_embedding.pkl'.format(args.data, args.model), 'wb') as f:
            pickle.dump(res, f)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, help='train or predict',
                        required=False, default='train')
    parser.add_argument('--clear_model', type=int, help='Whether to clear existing model',
                        required=False, default=1)
    parser.add_argument('--model', type=str, help='models: [skip_thought|quick_thought]',
                        required=False, default='transformer')
    parser.add_argument('--data', type=str, help='which data to use[data should be list of tokenized string]',
                        required=False, default='squad')
    parser.add_argument('--gpu', type=int, help='Whether to enable gpu',
                        required=False, default=0)
    args = parser.parse_args()

    main(args)


class args:
    clear_model=0
    gpu=0
    data='squad'
    model='transformer'
    step='predict'