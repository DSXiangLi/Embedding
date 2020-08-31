from BaseDataset import *
from config.quora_config import QUORA_PROTO
from config.default_config import *

class FasttextDataset( BaseDataset ):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count, invalid_index, padded_shape, padding_values, ngram):
        super( FasttextDataset, self ).__init__( data_file, dict_file, epochs, batch_size, buffer_size, min_count, invalid_index)
        self.padded_shape = padded_shape
        self.padding_values = padding_values
        self.ngram = ngram

    @staticmethod
    def parse_example(ngram):
        def helper(line):
            features = tf.parse_single_example(line, features = QUORA_PROTO)

            if ngram == 1:
                features['tokens'] = tf.sparse_tensor_to_dense(features['tokens'], default_value='')
                features.pop('ngram_tokens')
            else:
                features['tokens'] = tf.sparse_tensor_to_dense(features['ngram_tokens'], default_value='')
                features.pop( 'ngram_tokens' )
            target = tf.reshape(tf.cast( features.pop('target'), tf.float32),[-1]) # label dimension (,1)

            return features, target
        return helper

    @staticmethod
    def word_table_lookup(word_table):
        def helper(features, label ):
            features['tokens'] = word_table.lookup( features['tokens'] )
            return features, label
        return helper

    def build_dataset(self, is_predict):
        """
        dataset is generated in following steps:
        1. sentence split into list of words
        2. map words to tokens
        3. filter out word below min_count
        3. padded batch to the max_length
        4. shuffle, repeat
        """
        def input_fn():
            word_table_lookup = FasttextDataset.word_table_lookup(self.build_wordtable())
            parser = FasttextDataset.parse_example(self.ngram)

            dataset = tf.data.TFRecordDataset( self.data_file). \
                map( lambda x: parser( x ) ). \
                map( lambda features, label: word_table_lookup( features, label ) ). \
                filter( lambda features, label: tf.greater( tf.size(
                            tf.boolean_mask( features['tokens'], tf.not_equal( features['tokens'], self.invalid_index ) ) ), 1) )
            if not is_predict:
                dataset = dataset \
                    .shuffle(self.buffer_size ) \
                    .repeat( self.epochs )

            dataset = dataset.\
                padded_batch( batch_size= self.batch_size,
                              padded_shapes = self.padded_shape,
                              padding_values= self.padding_values,
                              drop_remainder=True).\
                prefetch(tf.data.experimental.AUTOTUNE)

            return dataset
        return input_fn


if __name__ == '__main__':

    input_pipe = FasttextDataset(data_file = './data/quora/train_2gram.tfrecords',
                                 dict_file = './data/quora/2gram_dictionary.pkl', #use ngram dictinoary accordingly
                                 epochs = 10,
                                 batch_size =5,
                                 min_count = 2,
                                 buffer_size = 128,
                                 invalid_index = 0,
                                 padded_shape = ({'tokens': [None],
                                                 'extra_features': [3]}, [1]),
                                 padding_values = ({'tokens': INVALID_INDEX, 'extra_features':0.0}, 0.0),
                                 ngram = 2
                                 )
    input_pipe.build_dictionary()
    input_fn = input_pipe.build_dataset(False)
    dataset = input_fn()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    iterator = tf.data.make_initializable_iterator(dataset)
    sess.run(iterator.initializer)

    sess.run(iterator.get_next())