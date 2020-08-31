import tensorflow as tf
import os
import pandas as pd
from data.preprocess_util import *

class TFDump(object):
    def __init__(self, data_dir, file, const_dir, language, ngram):
        self.data_dir = data_dir
        self.file = file
        self.prep = StrUtils(os.path.join(const_dir, language), language)
        self.sentences = None
        self.target = None
        self.ngram = ngram

    def load_data(self):
        print( 'Reading Raw corpus in {}'.format( self.data_dir) )
        df = pd.read_csv(os.path.join(self.data_dir, self.file + '.csv'))
        self.sentences = df['question_text'].values
        self.target = df['target'].values

    def preprocess (self, multiprocess):
        print('String Preprocess and word cut ')
        self.sentences = self.prep.text_cleaning(self.sentences)
        if multiprocess:
            self.tokens = self.prep.multi_word_cut( self.sentences )
        else:
            self.tokens = self.prep.word_cut(self.sentences)
        self.ngram_tokens = self.prep.make_ngram(self.ngram, self.tokens)

    def dump_tokens(self):
        with open(os.path.join(data_dir, '{}gram_tokens.pkl'.format(self.ngram)), 'wb') as f:
            pickle.dump(self.ngram_tokens, f)
        with open(os.path.join(data_dir, 'single_tokens.pkl'), 'wb') as f:
            pickle.dump(self.tokens, f)

    def feature_engineering(self):
        """
        sentences components:
        - number of words
        - number of char
        - number of puncts
        - number of stop words
        """
        print('Feature Engineering...')
        n_word = [len( sen ) for sen in self.tokens]
        n_char = [len( sen ) for sen in self.sentences]
        n_puncts = [len( self.prep.re_puncts.findall( sen ) ) for sen in self.sentences ]

        self.extra_feature = [[n_word[i], n_char[i], n_puncts[i]] for i in range(len(n_word))]

    @staticmethod
    def int_feature(value):
        if not isinstance( value, list ):
            value = [value]
        return tf.train.Feature( int64_list=tf.train.Int64List( value = value ) )

    @staticmethod
    def string_feature(value):
        if not isinstance( value, list ):
            value = [value]
        return tf.train.Feature( bytes_list=tf.train.BytesList( value = [bytes(i, encoding='UTF-8') for i in value]) )

    @staticmethod
    def float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature( float_list=tf.train.FloatList( value = value))

    def dump_tfrecord(self):
        print('Dumping TFRecord...')
        with tf.python_io.TFRecordWriter(os.path.join(self.data_dir, self.file + '.tfrecords')) as writer:
            for i in range(len(self.sentences)):
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'tokens': TFDump.string_feature(self.tokens[i]),
                            'ngram_tokens': TFDump.string_feature(self.ngram_tokens[i]),
                            'target': TFDump.float_feature(self.target[i]),
                            'extra_features': TFDump.float_feature(self.extra_feature[i])
                        }
                    )

                )

                writer.write(example.SerializeToString())

    def execute(self, multiprocess):
        self.load_data()
        self.preprocess(multiprocess)
        self.feature_engineering()
        self.dump_tfrecord()


if __name__ == '__main__':
    data_dir = 'data/quora'

    preprocess = TFDump(data_dir, 'train', 'const', 'en', 1)
    preprocess.execute(True)

    preprocess.dump_tokens() ##TODO: allow load data from mid-result and do extra feature engineering

    # Dump dictionary for single token & ngram
    dump_dictionary(data_dir, preprocess.tokens, '')
    dump_dictionary(data_dir, preprocess.ngram_tokens, '')
