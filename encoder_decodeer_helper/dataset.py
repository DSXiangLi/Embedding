# -*- coding=utf-8 -*-
import numpy as np
import tensorflow as tf
from BaseDataset import *


class Seq2SeqDataset(BaseDataset):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count, max_count,
                 special_token, max_len, min_len, pretrain_model_list):
        super(Seq2SeqDataset, self).__init__(data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                                             max_count, special_token)
        self.max_len = max_len
        self.min_len = min_len
        self.pretrain_model_list = pretrain_model_list
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
    def start_index(self):
        return self.special_mapping[self.special_token.SEQ_START]

    @property
    def end_index(self):
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
            decoder_source = self.make_source_dataset(self.data_file['decoder'], 'decoder', word_table_func)

            dataset = tf.data.Dataset.zip((encoder_source, decoder_source)).\
                filter(self.sample_filter_logic)

            if not is_predict:
                dataset = dataset. \
                    repeat()

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
            embedding_retriever = self.get_word_vector(self.pretrain_model_list)
            embedding = []
            for i in self._dictionary.keys():
                embedding.append(embedding_retriever(i))
            self.embedding = np.array(embedding, dtype=np.float32)
        return self.embedding

    @staticmethod
    def get_word_vector(word_vector_list):
        def helper(token):
            embedding = None
            for word_vector in word_vector_list:
                try:
                    embedding = word_vector.get_vector(token)
                except KeyError:
                    continue
            if embedding is None:
                embedding = np.random.uniform(low=-0.1, high=0.1, size=300)
            return embedding
        return helper


class BiSeq2SeqDataset(Seq2SeqDataset):
    """
    Bi-Lingual seq2seq Dataset, overwrite below for 2 language
    - build_dictionary
    - special_mapping
    """
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count, max_count,
                 special_token, max_len, min_len, pretrain_model_list):
        super(BiSeq2SeqDataset, self).__init__(data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                                               max_count, special_token, max_len, min_len, pretrain_model_list)

    def load_dictionary(self, filename):
        with open(filename, 'rb') as f:
            dictionary = pickle.load(f)
        dictionary = dict([(i, j) for i, j in dictionary.items() if (j >= self.min_count) and (j <= self.max_count)])
        return dictionary

    def build_dictionary(self):
        """
        Overwrite base, because it is bi-lingual, id order must be decoder < special < encoder, so that decoder only need
        first decoder+special size for loss calculation
        """
        encoder_dict = self.load_dictionary(self.dict_file['encoder'])
        decoder_dict = self.load_dictionary(self.dict_file['decoder'])

        self.decoder_size = len(decoder_dict)
        self.decoder_total_size = self.decoder_size + self.special_size

        encoder_list = [i for i in encoder_dict if i not in decoder_dict]
        dictionary= dict([(key,i) for i, key in enumerate(decoder_dict.keys())] + \
                          [(key, i + self.decoder_total_size) for i, key in enumerate(encoder_list)])
        dictionary.update(self.special_mapping)

        self._dictionary = collections.OrderedDict( sorted(dictionary.items(), key = lambda x:x[1]) )

    @property
    def special_mapping(self):
        return dict([(key, i + self.decoder_size) for i, key in enumerate(self.special_token._asdict().values())])


if __name__ == '__main__':
    from config.bookcorpus_config import MySpecialToken
    from config.default_config import get_pretrain_model
    sess = tf.Session()

    data = 'wmt'
    input_pipe = BiSeq2SeqDataset(data_file={
                                    'encoder': './data/{}/train_encoder_source.txt'.format(data),
                                    'decoder': './data/{}/train_decoder_source.txt'.format(data)
                                },
                                dict_file= {
                                    'encoder': './data/{}/encoder_dictionary.pkl'.format(data),
                                    'decoder': './data/{}/decoder_dictionary.pkl'.format(data),

                                },
                                epochs=10,
                                batch_size=1,
                                buffer_size=128,
                                max_len=20,
                                min_len=5,
                                min_count=3,
                                max_count=8000,
                                special_token=MySpecialToken,
                                pretrain_model_list=[get_pretrain_model('gn300')]
    )
    input_pipe.build_dictionary()

    print('Number of special token = {}'.format(input_pipe.special_size))
    print('Number of vocab = {}'.format(input_pipe.vocab_size ))
    print('Number of token vocab = {}'.format(input_pipe.total_size))
    print('Number of special mapping ={}'.format(input_pipe.special_mapping))

    input_fn = input_pipe.build_dataset(is_predict=False)
    dataset = input_fn()

    iterator = tf.data.make_initializable_iterator( dataset )
    sess.run( iterator.initializer )
    sess.run( tf.tables_initializer() )
    sess.run( tf.global_variables_initializer() )
    print(sess.run( iterator.get_next() ))

    embedding = input_pipe.load_pretrain_embedding()



