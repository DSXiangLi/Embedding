# -*- coding=utf-8 -*-
"""

Test RNN/GRU/LSTM rnn cell_size, dynamic_rnn state size for bridging
"""
import tensorflow as tf
import numpy as np
from skip_thought.seq2seq_utils import build_rnn_cell

X=np.random.randn(2, 3, 4)
X_lengths=[3, 2]
print(X.shape) # batch=2, seq_len=3, emb_size =4

# 1 gru
tf.reset_default_graph()
cell = build_rnn_cell('gru', {'hidden_units': [10], 'cell_size':1})

outputs, state =tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
s = sess.run(state) # tuple of length =1: element shape = batch * hidden = 2 * 10
sess.run(tf.shape(state)) # [1,2,10]
cell.state_size # (10,)

# 2 gru
tf.reset_default_graph()
cell = build_rnn_cell('gru', {'hidden_units': [10, 10], 'cell_size':2})

outputs, state =tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
s = sess.run(state) # tuple of length =2: element shape = batch * hidden = 2 * 10
sess.run(tf.shape(state)) # [2, 2, 10]
cell.state_size # (10,10)

# 1 lstm
tf.reset_default_graph()
cell = build_rnn_cell('lstm', {'hidden_units': [10], 'cell_size': 1})
outputs, state =tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
s = sess.run(state) # tuple of length =1: element is namedtuple LSTMStateTuple=(c,h)
s[0].h # each has shape = batch * hidden = 2 * 10
s[0].c
cell.state_size # tuple of length =1,


# 2 lstm
tf.reset_default_graph()
cell = build_rnn_cell('lstm', {'hidden_units': [10,10], 'cell_size': 2})
outputs, state =tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
s = sess.run(state) # tuple of length =1: element is namedtuple LSTMStateTuple=(c,h)
s[0].h # each has shape = batch * hidden = 2 * 10
s[0].c
cell.state_size # tuple of length =2, element = LSTMStateTuple(c=10, h=10)

paper/An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation.pdf
paper/How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.pdf
paper/Understanding Word2Vec and Paragraph2Vec.pdf
paper/[quick-thought] QuickThought_AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS.pdf
paper/[skipthouguht] skip thoughts.pdf
paper/How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.pdf