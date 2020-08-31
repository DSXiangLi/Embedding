import collections
import tensorflow as tf
import pickle

import os
class BaseDataset(object):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count, invalid_index):
        self.data_file = data_file
        self.dict_file = dict_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_count = min_count
        self._dictionary = None
        self.word_index = None
        self.invalid_index = invalid_index

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def vocab_size(self):
        return (len(self._dictionary) + 1)  # Leave place for 0 INVALID_INDEX

    def build_dictionary(self):
        """
        dictionray: {char:frequency} order by descending frequency
        word_index: {char:index} index is the above dictionary order
        """
        # raw dictionary built when preprocessing data
        with open(self.dict_file, 'rb') as f:
            dictionary = pickle.load(f)

        self._org_vocab_size = len(dictionary)

        if self.min_count >0:
            dictionary = dict([(i,j) for i,j in dictionary.items()  if j >= self.min_count  ])

        self._dictionary = collections.OrderedDict( sorted(dictionary.items(), key = lambda x:x[1], reverse = True) )

    def build_wordtable(self):
        # word_frequency < self.min_count will be map to INVALID_INDEX
        with tf.name_scope('wordtable'):
            return tf.lookup.StaticHashTable(
                initializer = tf.lookup.KeyValueTensorInitializer(
                    keys = list(self._dictionary.keys()),
                    values = list(range(1, (len(self._dictionary)+1) )), # leaving 0 for INVALID_INDEX
                    key_dtype = tf.string,
                    value_dtype = tf.int32
                ), default_value = self.invalid_index # all out of vocabulary will be map to INVALID_INDEX
            )


