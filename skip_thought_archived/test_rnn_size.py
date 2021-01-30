# -*- coding=utf-8 -*-
"""

Test RNN/GRU/LSTM rnn cell_size, dynamic_rnn state size for bridging
"""
import tensorflow as tf
import numpy as np
from skip_thought_archived.seq2seq_utils import build_rnn_cell

X=np.random.randn(2, 3, 4)
X_lengths=[3, 2]
print(X.shape) # batch=2, seq_len=3, emb_size =4

# 1 gru
tf.reset_default_graph()
cell = build_rnn_cell('gru', {'hidden_units': [10], 'cell_size':1, 'keep_prob':[0.1]})

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
