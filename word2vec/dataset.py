from BaseDataset import *


class Word2VecDataset(BaseDataset):
    def __init__(self, data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                 special_token, window_size, sample_rate, model):
        super(Word2VecDataset, self).__init__(data_file, dict_file, epochs, batch_size, buffer_size, min_count,
                                              special_token)
        self.model = model
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.param_check()

    def param_check(self):
        assert self.model in ['CBOW', 'SG'], 'For moedl only [CBOW | SG] are supported'

    def build_sampletable(self):
        logging.info('Building Sample table')

        sample_table = tf.lookup.StaticHashTable(
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys =  list(range(self.vocab_size)),
                values = [ 1- (self.sample_rate * self.vocab_size/i) **0.5 for i in self.vocab_dictionary.values() ],
                key_dtype = tf.int32,
                value_dtype = tf.float32,
            ), default_value = 1 # Don't sample Special Token
            , name = 'sample_table'
        )

        return sample_table

    @staticmethod
    def subsample(tokens, sampletable):
        return tf.boolean_mask(tensor = tokens,
                               mask = tf.less(sampletable.lookup(tokens),
                                              tf.random_uniform(shape = [tf.size(tokens)], minval = 0, maxval =1)))

    def window_slice_func(self):
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
                left, right = context_position(step, self.window_size, sentence_length)
                context_words = tf.gather( tokens, tf.concat( [left, right], axis=0 ) )  # (2* window_size,)
                center_word = tf.gather( tokens, [step] )  #(1,)

                # generate (feature,label) pair
                if self.model == 'CBOW':
                    label = tf.expand_dims( center_word, axis=0 ) # (1,1)
                    feature = tf.expand_dims(
                        tf.pad( context_words, paddings=[[0, 2 * self.window_size - tf.size( context_words )]],
                                mode='CONSTANT', constant_values= self.pad_index), axis=0 )  # (1, 2*window_size)
                else:
                    label = tf.expand_dims( context_words, axis=1 )  # (2* window_size,1)
                    feature = tf.fill( dims= tf.shape(context_words), value=center_word[0] )  # (2* window_size,  )

                return step + 1, features.write( step, feature ), labels.write( step, label )

            # Initialize variable
            step = tf.constant( 0 )
            features = tf.TensorArray( dtype=tf.int32, size=sentence_length, infer_shape=False )
            labels = tf.TensorArray( dtype=tf.int32, size=sentence_length, infer_shape=False )

            # run window sliding through sentences(list of tokens)
            _, features, labels = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[step, features, labels],
                back_prop=False
            )

            return features.concat(), labels.concat()

        return helper

    @staticmethod
    def parse_example(line):

        tokens = tf.string_split( [tf.string_strip( line )] ).values

        return tokens

    @staticmethod
    def sample_filter_logic(line):

        filter = tf.greater(tf.size(line), 2)

        return filter

    def build_dataset(self, is_predict=0):
        """
        dataset is generated in following steps:
        1. sentence split into list of words
        2. map words to tokens
        3. filter out word below min_count, subsample vocab given sample rate
        3. apply window slicing of each sentences, every token is label, features are padded to same length
        4. shuffle, repeat, batch
        """

        def input_fn():
            wordtable = self.build_wordtable() # must be init after estimator, because estimator creates new graph
            sampletable = self.build_sampletable()
            window_slice_func = self.window_slice_func()

            dataset = tf.data.TextLineDataset(self.data_file).\
                        map(lambda x: Word2VecDataset.parse_example(x)). \
                        map(lambda x: wordtable.lookup(x)). \
                        map(lambda x: Word2VecDataset.subsample(x, sampletable)). \
                        filter(Word2VecDataset.sample_filter_logic).\
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
    sess = tf.Session()
    from config.sogou_news_config import MySpecialToken
    input_pipe = Word2VecDataset(data_file = './data/sogou_news/corpus_new.txt',
                                 dict_file = './data/sogou_news/dictionary.pkl',
                                 epochs = 10,
                                 batch_size =5,
                                 min_count = 2,
                                 special_token=MySpecialToken,
                                 buffer_size = 128,
                                 window_size=2,
                                 sample_rate=0.01,
                                 model='CBOW'
                                 )

    input_pipe.build_dictionary()
    input_fn = input_pipe.build_dataset()
    dataset = input_fn()

    iterator = tf.data.make_initializable_iterator(dataset)
    sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.get_next() )
