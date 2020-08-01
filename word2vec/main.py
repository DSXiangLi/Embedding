import tensorflow as tf
import argparse
from utils import clear_model, build_estimator
from dataset import Word2VecDataset
from model import model_fn
from config.default_config import CHECKPOINT_DIR, TRAIN_PARAMS

#import sys
#sys.path.append('/Users/xiangli/Desktop/Embedding/word2vec')

def main(args):

    model_dir = CHECKPOINT_DIR.format(args.model, args.train_algo)
    datafile = './data/{}/corpus_new.txt'.format(args.data)

    if args.clear_model:
        clear_model(model_dir)

    # Init dataset
    input_pipe = Word2VecDataset( filename= datafile,
                                  model= args.model,
                                  window_size= TRAIN_PARAMS['window_size'],
                                  epochs= TRAIN_PARAMS['epochs'],
                                  batch_size=TRAIN_PARAMS['batch_size'],
                                  buffer_size=TRAIN_PARAMS['buffer_size'],
                                  min_count=TRAIN_PARAMS['min_count'],
                                  sample_rate = TRAIN_PARAMS['sample_rate'])

    input_pipe.build_dictionary()

    TRAIN_PARAMS.update(
        {
            'vocab_size': input_pipe.vocab_size,
            'freq_dict': input_pipe.dictionary,
            'train_algo': args.train_algo,
            'loss': args.loss,
            'model': args.model
        }
    )

    # Init Estimator
    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn)

    train_spec = tf.estimator.TrainSpec( input_fn = input_pipe.build_dataset() )

    eval_spec = tf.estimator.EvalSpec( input_fn = input_pipe.build_dataset(is_predict=1),
                                       steps=200,
                                       throttle_secs=60 )

    tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec )


if __name__ == '__main__':
    tf.logging.set_verbosity( tf.logging.ERROR )
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type = str, help = 'SG=skip gram, CBOW=Continuous Bag of Words', required=True )
    parser.add_argument( '--train_algo', type = str, help = 'NG=negative sampling, HS=Hierarchy Softamx', required=True)
    parser.add_argument( '--loss', type=str, help='nce_loss or sample_loss or "" for Hierarchy Softmax', required=False, default = '' )
    parser.add_argument( '--step', type=str, help='train or predict', required=False, default = 'train' )
    parser.add_argument( '--clear_model', type=int, help= 'Whether to clear existing model', required=False, default=1)
    parser.add_argument( '--data', type=str, help='which data to use[data should be list of tokenized string]', required=False, default='sogou_news')
    args = parser.parse_args()

    main(args)
