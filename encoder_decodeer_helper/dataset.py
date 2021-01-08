# -*- coding=utf-8 -*-

import numpy as np
import tensorflow as tf
from gensim import models

from BaseDataset import *


class Seq2SeqDataset(BaseDataset):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count, max_count,
                 special_token, max_len, min_len, pretrain_model):
        super(Seq2SeqDataset, self).__init__(data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                                             max_count, special_token)
        self.max_len = max_len
        self.min_len = min_len
        self.pretrain_model = pretrain_model
        self.embedding = None
        self.params_check()

    def params_check(self):
        """
        what to check
        1. whether speical token has <PAD>, <UNK>, <SEQ_START>, <SEQ_END>
        """
        assert all([(i in self.special_token._fields) for i in ['PAD', 'UNK', 'SEQ_START', 'SEQ_END']]), \
            'For Encoder-Decoder special token must have PAD, UNK, SEQ_START, SEQ_END'
        assert all([i in self.data_file.keys() for i in ['encoder', 'decoder']]), \
            'data file must specify encoder and decoder'

    def parse_example(self, line, prepend, append):
        """
        Input:
            line: line of text string
            prepend: whether to add sequence start
            append: wheteher to add sequence end
        Return:
            feature: {tokens:, seq_len:}
        """
        features = {}
        tokens = tf.string_split([tf.string_strip(line)]).values

        if prepend:
            tokens = tf.concat([[self.special_token.SEQ_START], tokens], 0)
        if append:
            tokens = tf.concat([tokens, [self.special_token.SEQ_END]], 0)

        features['tokens'] = tokens
        features['seq_len'] = tf.size(tokens)
        return features

    @property
    def start_token(self):
        return self.special_mapping[self.special_token.SEQ_START]

    @property
    def end_token(self):
        return self.special_mapping[self.special_token.SEQ_END]

    @staticmethod
    def word_table_lookup(word_table):
        def helper(features):
            features['tokens'] = word_table.lookup( features['tokens'] )
            return features
        return helper

    @staticmethod
    def prepend_append_logic(data_type):
        if data_type == 'encoder':
            prepend = False
            append = False
        elif data_type == 'decoder':
            # decodere_source in train&eval, start_token used in train, end_token used in eval
            prepend = True
            append = True
        else:
            raise Exception('data_type {} can only be [encoder/ decoder]'.format(data_type) )
        return prepend, append

    def sample_filter_logic(self, encoder_source, decoder_source):
        """
        Filter sample with length <=1 or length >= max_length. Filter must be applied after zip,
        otherwise encoder and decoder data will mis match
        """
        filter_encoder = tf.logical_and(tf.greater(encoder_source['seq_len'], self.min_len),
                                        tf.less(encoder_source['seq_len'], self.max_len))
        filter_decoder = tf.logical_and(tf.greater(decoder_source['seq_len'], self.min_len),
                                        tf.less(decoder_source['seq_len'], self.max_len))

        return tf.logical_and(filter_encoder, filter_decoder)

    def make_source_dataset(self, file_path, data_type, word_table_func):
        """
        Build datast for encoder/decoder
        1. read in text file
        2. parse_example and add token
        2. convert word to token
        """
        prepend, append = self.prepend_append_logic(data_type)

        dataset = tf.data.TextLineDataset(file_path).\
            map(lambda x: self.parse_example(x, prepend, append), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
            map(lambda x: word_table_func(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    @property
    def padded_shape(self):
        return ({'tokens': [None], 'seq_len': []},
                {'tokens': [None], 'seq_len': []})

    @property
    def padding_values(self):
        """
        padding value for sequence is the id for <PAD>
        """
        return ({'tokens': self.pad_index, 'seq_len': 0},
                {'tokens': self.pad_index, 'seq_len': 0})

    def build_dataset(self, is_predict=0):
        """
        Bulid Encoder-Decoder Dataset
        1. make encoder, decoder dataset independently
        2. zip to create feature and labels
        3. filter sample outside length range
        4. repeat & shuffle & padded_batch
        """
        def input_fn():
            word_table_func = self.word_table_lookup(self.build_wordtable())
            _ = self.build_tokentable() # initialize here to ensure lookup table is in the same graph

            encoder_source = self.make_source_dataset(self.data_file['encoder'], 'encoder', word_table_func)
            if is_predict:
                # for prediction no decoder source is needed
                data_file = self.data_file['encoder']
            else:
                data_file = self.data_file['decoder']
            ##TODO: eval use decoder source input but can't use teacher forcing in evaluation
            decoder_source = self.make_source_dataset(data_file, 'decoder', word_table_func)

            dataset = tf.data.Dataset.zip((encoder_source, decoder_source)).\
                filter(self.sample_filter_logic)

            if not is_predict:
                dataset = dataset.\
                    repeat(self.epochs)

                dataset = dataset. \
                    padded_batch( batch_size=self.batch_size,
                                  padded_shapes=self.padded_shape,
                                  padding_values=self.padding_values,
                                  drop_remainder=True ). \
                    prefetch( tf.data.experimental.AUTOTUNE )
            else:
                dataset = dataset.batch(1)

            return dataset
        return input_fn

    def load_pretrain_embedding(self):
        if self.embedding is None:
            word_vector = models.KeyedVectors.load_word2vec_format(self.pretrain_model, binary=True)
            embedding = []
            for i in self._dictionary.keys():
                try:
                    embedding.append(word_vector.get_vector(i))
                except KeyError:
                    embedding.append(np.random.uniform(low=-0.1, high=0.1, size=300))
            self.embedding = np.array(embedding, dtype=np.float32)
        return self.embedding


if __name__ == '__main__':
    from config.squad_config import MySpecialToken, TRAIN_PARAMS
    sess = tf.Session()

    data = 'squad'
    input_pipe = Seq2SeqDataset(data_file={
                                    'encoder': './data/{}/train_encoder_source.txt'.format(data),
                                    'decoder': './data/{}/train_decoder_source.txt'.format(data)
                                },
                                dict_file='./data/{}/dictionary.pkl'.format(data),
                                epochs=TRAIN_PARAMS['epochs'],
                                batch_size=TRAIN_PARAMS['batch_size'],
                                buffer_size=TRAIN_PARAMS['buffer_size'],
                                min_count=TRAIN_PARAMS['min_count'],
                                max_count=TRAIN_PARAMS['max_count'],
                                special_token=MySpecialToken,
                                max_len=TRAIN_PARAMS['max_decode_iter'],
                                min_len=TRAIN_PARAMS['min_len'],
                                pretrain_model=TRAIN_PARAMS['pretrain_model']
    )
    input_pipe.build_dictionary()

    print('Number of special token = {}'.format(input_pipe.special_size))
    print('Number of vocab = {}'.format(input_pipe.vocab_size ))
    print('Number of token vocab = {}'.format(input_pipe.total_size))
    print('Number of special mapping ={}'.format(input_pipe.special_mapping))

    input_fn = input_pipe.build_dataset()
    dataset = input_fn()

    iterator = tf.data.make_initializable_iterator( dataset )
    sess.run( iterator.initializer )
    sess.run( tf.tables_initializer() )
    sess.run( tf.global_variables_initializer() )
    print(sess.run( iterator.get_next() ))

    input_pipe.load_pretrain_embedding()
    print(input_pipe.embedding[0])


