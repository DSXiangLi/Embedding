
import argparse
import importlib
import pickle
from utils import clear_model, build_estimator
from encoder_decodeer_helper.dataset import Seq2SeqDataset
from config.default_config import CHECKPOINT_DIR, DICTIONARY_DIR
from cnn_lstm.model import model_fn
from cnn_lstm.myconfig import ALL_TRAIN_PARAMS, ALL_RUN_CONFIG

MODEL = 'cnn_lstm'

def main(args):
    ## Init directory
    model_dir = CHECKPOINT_DIR.format(args.data, MODEL)
    dict_file = DICTIONARY_DIR.format(args.data)

    # Init config
    TRAIN_PARAMS = ALL_TRAIN_PARAMS[args.data]
    RUN_CONFIG = ALL_RUN_CONFIG[args.data]
    MySpecialToken = getattr(importlib.import_module('config.{}_config'.format(args.data)), 'MySpecialToken')

    if args.step == 'train':
        data_file = {
            'encoder': './data/{}/train_encoder_source.txt'.format(args.data),
            'decoder': './data/{}/train_decoder_source.txt'.format(args.data)
        }
    else:
        data_file = {
            'encoder': './data/{}/dev_encoder_source.txt'.format(args.data),
            'decoder': './data/{}/dev_decoder_source.txt'.format(args.data) # for predict, this can be same as encoder
        }

    if args.clear_model:
        clear_model(model_dir)

    # Init dataset
    input_pipe = Seq2SeqDataset(data_file=data_file,
                                dict_file=dict_file,
                                epochs=TRAIN_PARAMS['epochs'],
                                batch_size=TRAIN_PARAMS['batch_size'],
                                min_count=TRAIN_PARAMS['min_count'],
                                max_count=TRAIN_PARAMS['max_count'],
                                buffer_size=TRAIN_PARAMS['buffer_size'],
                                special_token=MySpecialToken,
                                max_len=TRAIN_PARAMS['max_len'],
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
            'start_index': input_pipe.start_index,
            'end_index': input_pipe.end_index,
            'pretrain_embedding': input_pipe.load_pretrain_embedding()
        }
    )

    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn, args.gpu, RUN_CONFIG)

    if args.step == 'train':
        estimator.train(input_fn=input_pipe.build_dataset())

    if args.step == 'predict':
        # Please disable GPU in prediction to avoid DST exhausted Error
        prediction = estimator.predict(input_fn=input_pipe.build_dataset(is_predict=1))
        res = []
        for i in prediction:
            res.append(i)
        with open('./data/{}/{}_predict.pkl'.format(args.data, MODEL), 'wb') as f:
            pickle.dump(res, f)


if __name__ == '__main__':

    import logging
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, help='train or predict',
                        required=False, default='train')
    parser.add_argument('--clear_model', type=int, help='Whether to clear existing model',
                        required=False, default=0)
    parser.add_argument('--data', type=str, help='which data to use[data should be list of tokenized string]',
                        required=False, default='bookcorpus')
    parser.add_argument('--gpu', type=int, help='Whether to enable gpu',
                        required=False, default=0)
    args = parser.parse_args()

    main(args)
