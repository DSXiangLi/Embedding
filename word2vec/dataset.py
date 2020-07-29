import tensorflow as tf
import collections
from config.default_config import INVALID_INDEX

class Word2VecDataset(object):
    def __init__(self, filename, model, window_size, epochs, batch_size, buffer_size ):
        self.filename = filename
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.window_size = window_size
        self._dictionary = None
        self.word_index = None

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def vocab_size(self):
        return len(self._dictionary)

    def build_dictionary(self):
        """
        dictionray: {char:frequency} order by descending frequency
        word_index: {char:index} index is the above dictionary order
        """
        dictionary = collections.Counter()
        with open(self.filename, 'r') as f:
            for line in f:
                dictionary.update(line.strip().split())

        self._dictionary = collections.OrderedDict( sorted(dictionary.items(), key = lambda x:x[1], reverse = False) )

    def build_wordtable(self):

        with tf.name_scope('wordtable'):
            return tf.lookup.StaticVocabularyTable(
                initializer = tf.lookup.KeyValueTensorInitializer(
                    keys = list(self._dictionary.keys()),
                    values = list(range(len(self._dictionary))),
                    key_dtype = tf.string,
                    value_dtype = tf.int64
                ), num_oov_buckets = 1
            )

    @staticmethod
    def window_slice_func(model, window_size):
        """
        Return
        CBOW: one instance, features = (2*window_size, ), labels = (1,1)
        Skip-Gram: 2*window instances, features= (1,) labels = (1,1)
        Note. labels needed to be 2-dimension required by sampled_loss, features must be 1-dimension to avoid extra
        dimension in embedding_lookup
        """

        def helper(tokens):

            sentence_length = tf.size( tokens )

            def cond(step, features, labels):
                return step < sentence_length

            def context_position(step, window_size, sentence_length):
                # get all context in window: feature for CBOW, targets for SG
                left = tf.range( start=tf.maximum( tf.constant( 0, dtype=tf.int32 ), step - window_size ),
                                 limit=step )
                right = tf.range( start=tf.minimum( sentence_length, step + 1 ),
                                  limit=tf.minimum( sentence_length, step + 1 + window_size ) )
                return left, right

            def body(step, features, labels):
                left, right = context_position(step, window_size, sentence_length)
                context_words = tf.gather( tokens, tf.concat( [left, right], axis=0 ) )  # (2* window_size,)
                center_word = tf.gather( tokens, [step] )  #(1,)

                # generate (feature,label) pair
                if model == 'CBOW':
                    label = tf.expand_dims( center_word, axis=0 ) # (1,1)
                    feature = tf.expand_dims(
                        tf.pad( context_words, paddings=[[0, 2 * window_size - tf.size( context_words )]],
                                mode='CONSTANT', constant_values= INVALID_INDEX ), axis=0 )  # (1, 2*window_size)
                else:
                    label = tf.expand_dims( context_words, axis=1 )  # (2* window_size,1)
                    feature = tf.fill( dims= tf.shape(context_words), value=center_word[0] )  # (2* window_size,  )

                return step + 1, features.write( step, feature ), labels.write( step, label )

            # Initialize variable
            step = tf.constant( 0 )
            features = tf.TensorArray( dtype=tf.int64, size=sentence_length, infer_shape=False )
            labels = tf.TensorArray( dtype=tf.int64, size=sentence_length, infer_shape=False )

            # run window sliding through sentences(list of tokens)
            _, features, labels = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[step, features, labels],
                back_prop=False
            )

            return features.concat(), labels.concat()

        return helper

    def build_dataset(self, is_predict=0):
        """
        dataset is generated in following steps:
        1. sentence split into list of words
        2. map words to tokens
        3. apply window slicing of each sentences, every token is label, features are padded to same length

        4. shuffle, repeat, batch
        """

        def input_fn():
            wordtable = self.build_wordtable() # must be init after estimator, because estimator creates new graph
            window_slice_func = Word2VecDataset.window_slice_func( self.model, self.window_size )

            dataset = tf.data.TextLineDataset(self.filename).\
                        map(lambda x: tf.string_split([x]).values).\
                        filter(lambda x: tf.greater(tf.size(x), 2)).\
                        map(lambda x: wordtable.lookup(x)).\
                        map( lambda x: window_slice_func( x ) ).\
                        flat_map( lambda features, label: tf.data.Dataset.from_tensor_slices( (features, label) ) )

            if not is_predict:
                dataset = dataset \
                    .shuffle(self.buffer_size ) \
                    .repeat( self.epochs )

            dataset = dataset.batch(self.batch_size, drop_remainder=True) # enable batch_size unstack

            return dataset
        return input_fn

if __name__ == '__main__':
    # test
    input_pipe = Word2VecDataset(filename = './data/sogou_news/corpus_new.txt',
                                 model = 'CBOW',
                                 window_size = 2,
                                 epochs = 10,
                                 batch_size =5,
                                 buffer_size = 128)

    input_pipe.build_dictionary()
    print(input_pipe.dictionary)

    func  = input_pipe.build_dataset()
    dataset = func()

    sess = tf.Session()
    iterator = tf.data.make_initializable_iterator(dataset)

    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    features, labels = sess.run(iterator.get_next() )
    print(features)
    print(labels)