import heapq
import tensorflow as tf
from config.default_config import MyWordSpecialToken


class TreeNode(object):
    total_node = 0

    def __init__(self, frequency, char = None , word_index = None, is_leaf = False):
        self.frequency = frequency
        self.char = char # word character
        self.word_index = word_index # word look up index
        self.left = None
        self.right = None
        self.is_leaf = is_leaf
        self.counter(is_leaf)

    def counter(self, is_leaf):
        # node_index will be used for embeeding_lookup
        self.node_index = TreeNode.total_node
        if not is_leaf: TreeNode.total_node += 1

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __repr__(self):
        if self.is_leaf:
            return 'Leaf Node char = [{}] index = {} freq = {}'.format(self.char, self.word_index, self.frequency)
        else:
            return 'Inner Node [{}] freq = {}'.format(self.node_index, self.frequency)


class HuffmanTree(object):
    def __init__(self, freq_dic, pad_index):
        self.nodes = []
        self.root = None
        self.max_depth = None
        self.freq_dic = freq_dic
        self.pad_index = pad_index
        self.all_paths = {}
        self.all_codes = {}
        self.node_index = 0

    @staticmethod
    def merge_node(left, right):
        parent = TreeNode(left.frequency + right.frequency)
        parent.left = left
        parent.right = right
        return parent

    def build_tree(self):
        """
        Build huffman tree with word being leaves
        """
        TreeNode.total_node = 0 # avoid train_and_evaluate has different node_index

        heap_nodes = []
        for word_index, (char, freq) in enumerate(self.freq_dic.items()):
            tmp = TreeNode( freq, char, word_index+1, is_leaf=True ) # add +1 because 0 is left for 0 INVALID_INDEX
            heapq.heappush(heap_nodes, tmp )

        while len(heap_nodes)>1:
            node1 = heapq.heappop(heap_nodes)
            node2 = heapq.heappop(heap_nodes)
            heapq.heappush(heap_nodes, HuffmanTree.merge_node(node1, node2))

        self.root = heapq.heappop(heap_nodes)

    @property
    def num_node(self):
        return self.root.node_index + 1

    def traverse(self):
        """
        Compute all node to leaf path and direction: list of node_id, list of 0/1
        """
        def dfs_helper(root, path, code):
            if root.is_leaf :
                self.all_paths[root.word_index] = path
                self.all_codes[root.word_index] = code
                return
            if root.left :
                dfs_helper(root.left, path + [root.node_index], code + [0])
            if root.right :
                dfs_helper(root.right, path + [root.node_index], code + [1])

        dfs_helper(self.root, [], [] )

        self.max_depth = max([len(i) for i in self.all_codes.values()])


class HierarchySoftmax(HuffmanTree):
    def __init__(self, freq_dic, pad_index):
        super(HierarchySoftmax, self).__init__(freq_dic, pad_index)

    def convert2tensor(self):
        # padded to max_depth and convert to tensor
        with tf.name_scope('hstree_code'):
            self.code_table = tf.convert_to_tensor([ code + [self.pad_index] * (self.max_depth - len(code)) for word, code
                                                     in sorted( self.all_codes.items(),  key=lambda x: x[0] )],
                                                   dtype = tf.float32)
        with tf.name_scope('hstree_path'):
            self.path_table = tf.convert_to_tensor([path + [self.pad_index] * (self.max_depth - len(path)) for word, path
                                                    in sorted( self.all_paths.items(), key=lambda x: x[0] )],
                                                   dtype = tf.int32)

    def get_loss(self, input_embedding_vector, labels, output_embedding, output_bias, params):
        """
        :param input_embedding_vector: [batch * emb_size]
        :param labels: word index [batch * 1]
        :param output_embedding: entire embedding matrix []
        :return:
            loss
        """
        loss = []
        labels = tf.unstack(labels, num = params['batch_size']) # list of [1]
        inputs = tf.unstack(input_embedding_vector, num = params['batch_size']) # list of [emb_size]

        for label, input in zip(labels, inputs):

            path = self.path_table[tf.squeeze(label)]#  (max_depth,)
            code = self.code_table[tf.squeeze(label)] # (max_depth,)

            # Mask padded value
            path = tf.boolean_mask(path, tf.not_equal(path, self.pad_index)) # (real_path_length,)
            code = tf.boolean_mask(code, tf.not_equal(code, self.pad_index) ) # (real_path_length,)

            output_embedding_vector = tf.nn.embedding_lookup(output_embedding, path) # real_path_length * emb_size
            bias = tf.nn.embedding_lookup(output_bias, path) # (real_path_length,)

            logits = tf.matmul(tf.expand_dims(input, axis=0), tf.transpose(output_embedding_vector) ) + bias # (1,emb_size) *(emb_size, real_path_length)
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels = code, logits = tf.squeeze(logits) ))

        loss = tf.reduce_mean(tf.concat(loss, axis = 0), axis=0, name = 'hierarchy_softmax_loss') # batch -> scaler

        return loss


if __name__ == '__main__':
    from word2vec.dataset import Word2VecDataset
    input_pipe = Word2VecDataset(data_file = './data/sogou_news/corpus_new.txt',
                                 dict_file = './data/sogou_news/dictionary.pkl',
                                 epochs = 10,
                                 batch_size =5,
                                 min_count = 2,
                                 sample_rate = 0.01,
                                 special_token=MyWordSpecialToken,
                                 buffer_size = 128,
                                 model='CBOW',
                                 window_size=2
                                 )

    input_pipe.build_dictionary()
    print(input_pipe.dictionary)
    print(input_pipe.vocab_size)

    hstree = HierarchySoftmax(input_pipe.dictionary)
    hstree.build_tree()
    hstree.traverse()
    print(hstree.num_node)
    print(hstree.root)
    print(hstree.all_paths[0])
    print( hstree.all_codes[0])

    sess = tf.Session()
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    hstree.convert2tensor()
    print(sess.run(hstree.code_table[0]))
    print(sess.run(hstree.path_table[0]))