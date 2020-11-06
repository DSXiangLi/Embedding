import collections
import tensorflow as tf
import pickle
import logging


class BaseDataset(object):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count, special_token):
        """
        Input
            data_file: file path for dataset
            dict_file: file path for vocabulary dictionary(default to pickle dump)
            epochs, batch_size, buffer_size: train parameter to build input dataset pipeple
            min_count: minimal number of word occurence, smaller than this will be mapped to UNK
            speical_token: iterable containing all special character, eg. UNK, PAD, Sequence Start/end
        Return
            dictionary: vocab to count dictionary
            vocab_size: vocabulary size including special character
            speical_mapping: special character to special id mapping, id start from vocab_size
        """
        self.data_file = data_file
        self.dict_file = dict_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_count = min_count
        self._dictionary = None
        self.special_token = special_token
        self.word_table = None
        self.token_table = None

    def params_check(self):
        raise NotImplementedError()

    @property
    def dictionary(self):
        # Return entire dictionary including special token
        return self._dictionary

    @property
    def vocab_dictionary(self):
        # Return only vocab dictionary
        return collections.OrderedDict([(i,j) for i,j in self._dictionary.items() if j != -1])

    @property
    def special_size(self):
        return len(self.special_token)

    @property
    def vocab_size(self):
        return self.total_size - self.special_size

    @property
    def total_size(self):
        return len(self._dictionary)

    @property
    def special_mapping(self):
        return dict([(token, self.vocab_size + i) for i, token in enumerate(self.special_token)])

    @property
    def pad_index(self):
        return self.special_mapping[self.special_token.PAD]

    def build_dictionary(self):
        """
        _dictionray: {char:frequency} order by descending frequency
        1. filter min_count character
        2. append dictionary by special character, special character all have count = -1
        """
        # raw dictionary built when preprocessing data
        with open(self.dict_file, 'rb') as f:
            dictionary = pickle.load(f)

        if self.min_count>0:
            dictionary = dict([(i,j) for i,j in dictionary.items() if j >= self.min_count])

        dictionary.update(dict.fromkeys(self.special_token._asdict().values(), -1))

        self._dictionary = collections.OrderedDict( sorted(dictionary.items(), key = lambda x:x[1], reverse = True) )

    def build_wordtable(self):
        logging.info('Building word table')

        self.word_table = tf.lookup.StaticHashTable(
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys = list(self._dictionary.keys()),
                values = list(range(self.total_size)),
                key_dtype = tf.string,
                value_dtype = tf.int32
            ), default_value = self.special_mapping[self.special_token.UNK] # unseen vocab will be map to UNK id
            , name ='word_table'
        )

        tf.add_to_collection(self.word_table.name, self.word_table)

        return self.word_table

    def build_tokentable(self):
        logging.info('Building Token table')

        self.token_table = tf.lookup.StaticHashTable(
                initializer = tf.lookup.KeyValueTensorInitializer(
                    keys = list(range(self.total_size)),
                    values = list(self._dictionary.keys()),
                    key_dtype = tf.int32,
                    value_dtype = tf.string
                ), default_value = self.special_token.UNK # unseen token will be map to UNK
                , name = 'token_table'
            )

        tf.add_to_collection(self.token_table.name, self.token_table )

        return self.token_table

    def sample_filter_logic(self, *wargs):
        """
        Specific sample filter logic goes here
        """
        raise NotImplementedError()

    def build_dataset(self, is_predict=0):
        """
        Build input_fn for estimator, return dataset
        """
        raise NotImplementedError()
