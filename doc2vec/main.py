# -*- coding=utf-8 -*-

import argparse
from utils import clear_model
from config.doc2vec_config import *
from doc2vec.model import *


def main(args):

    model_dir = CHECKPOINT_DIR.format(args.model, args.data)
    data_path = './data/{}/corpus_new.txt'.format(args.data)

    if args.model == 'doc2vec':
        model = Doc2VecModel(model_dir = model_dir, data_path = data_path, params = DV_PARAMS)
    elif args.model == 'word2vec':
        model = Word2VecModel( model_dir = model_dir, data_path=data_path, params= WV_PARAMS )
    else:
        raise Exception('Only word2vec and doc2vec are supported')

    if args.continue_train:
        model.model_load()

    model.fit(continue_train = args.continue_train)
    model.model_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type =str, help = 'which model to use',
                        required=False, default ='doc2vec')
    parser.add_argument('--continue_train', type=int, help = 'Whether to continue training',
                        required=False, default=1)
    parser.add_argument( '--data', type=str, help='which data to use[data should be list of tokenized string]',
                         required=False, default='sogou_news_big')
    args = parser.parse_args()

    main(args)
