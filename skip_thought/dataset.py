# -*- coding=utf-8 -*-

from BaseDataset import *


class SkipThoughtDataset(BaseDataset):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                 special_token, max_len):
        super(SkipThoughtDataset, self).__init__(data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                                                 special_token)
        self.max_len = max_len
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
    def prepend_append_logic(data_type, is_predict):
        if data_type == 'encoder':
            prepend = False
            append = False
        elif data_type == 'decoder':
            if is_predict:
                # decoder_source is not used in infer, only in loss calculation, append is done in Helper
                prepend = False
                append = True
            else:
                # decodere_source in train&eval, start_token used in train, end_token used in eval
                prepend = True
                append = True
        else:
            raise Exception('data_type {} can only be [encoder/ decoder_input / decoder_target]'.format(data_type) )
        return prepend, append

    def sample_filter_logic(self, encoder_source, decoder_source):
        """
        Filter sample with length <=1 or length >= max_length. Filter must be applied after zip,
        otherwise encoder and decoder data will mis match
        """
        filter_encoder = tf.logical_and(tf.greater(encoder_source['seq_len'], 1),
                                        tf.less(encoder_source['seq_len'], self.max_len))
        filter_decoder = tf.logical_and(tf.greater(decoder_source['seq_len'], 1),
                                        tf.less(decoder_source['seq_len'], self.max_len))

        return tf.logical_and(filter_encoder, filter_decoder)

    def make_source_dataset(self, file_path, data_type, is_predict, word_table_func):
        """
        Build datast for encoder/decoder
        1. read in text file
        2. parse_example and add token
        2. convert word to token
        """
        prepend, append = self.prepend_append_logic(data_type, is_predict)

        dataset = tf.data.TextLineDataset(file_path).\
            map(lambda x: self.parse_example(x, prepend, append)).\
            map(lambda x: word_table_func(x))

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

            encoder_source = self.make_source_dataset(self.data_file['encoder'], 'encoder', is_predict, word_table_func)
            decoder_source = self.make_source_dataset(self.data_file['decoder'], 'decoder', is_predict, word_table_func)

            dataset = tf.data.Dataset.zip((encoder_source, decoder_source)).\
                filter(self.sample_filter_logic)

            if not is_predict:
                dataset = dataset.\
                    shuffle(self.buffer_size).\
                    repeat(self.epochs)

            dataset = dataset. \
                padded_batch( batch_size=self.batch_size,
                              padded_shapes=self.padded_shape,
                              padding_values=self.padding_values,
                              drop_remainder=True ). \
                prefetch( tf.data.experimental.AUTOTUNE )

            return dataset
        return input_fn


if __name__ == '__main__':
    from config.bookcorpus_config import MySpecialToken, TRAIN_PARAMS
    sess = tf.Session()

    input_pipe = SkipThoughtDataset(data_file={'encoder': './data/bookcorpus/encoder_source.txt',
                                               'decoder': './data/bookcorpus/decoder_source.txt'},
                                    dict_file='./data/bookcorpus/dictionary.pkl',
                                    epochs=TRAIN_PARAMS['epochs'],
                                    batch_size=TRAIN_PARAMS['batch_size'],
                                    buffer_size=TRAIN_PARAMS['buffer_size'],
                                    min_count=TRAIN_PARAMS['min_count'],
                                    special_token=MySpecialToken,
                                    max_len=TRAIN_PARAMS['max_decode_iter'])
    input_pipe.build_dictionary()

    print('Number of special token = {}'.format(input_pipe.special_size))
    print('Number of vocab = {}'.format(input_pipe.vocab_size ))
    print('Number of token vocab = {}'.format(input_pipe.total_size))
    print('Number of special mapping ={}'.format(input_pipe.special_mapping))

    input_fn = input_pipe.build_dataset(is_predict=1)
    dataset = input_fn()

    iterator = tf.data.make_initializable_iterator( dataset )
    sess.run( iterator.initializer )
    sess.run( tf.tables_initializer() )
    sess.run( tf.global_variables_initializer() )
    print(sess.run( iterator.get_next() ))
