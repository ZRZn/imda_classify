#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
import os
from attention import attention

from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.layers import fully_connected
from utils import *

NUM_EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_SIZE = 50
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 100
SEN_LENTH = 40
DOCU_LENTH = 10
KEEP_PROB = 0.8
DELTA = 0.5

#Load Data
train_fir = open("/Users/zrzn/Downloads/imda_classify/train.pkl", "rb")
test_fir = open("/Users/zrzn/Downloads/imda_classify/test.pkl", "rb")
train_X = pickle.load(train_fir)
train_Y = pickle.load(train_fir)
test_X = pickle.load(test_fir)
test_Y = pickle.load(test_fir)

train_fir.close()
test_fir.close()

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

#placeholders
words_data = tf.placeholder(tf.int32, [BATCH_SIZE, DOCU_LENTH, SEN_LENTH])
labels = tf.placeholder(tf.float32, [BATCH_SIZE])
keep_prob_ph = tf.placeholder(tf.float32)
# seq_len_ph = tf.placeholder(tf.int32, [None])
# docu_len_ph = tf.placeholder(tf.int32, [None])
#Embedding Layer
embeddings = tf.Variable(tf.random_uniform([50000, EMBEDDING_SIZE], -1.0, 1.0), trainable=True, name="embeddings")
word_embedded = tf.nn.embedding_lookup(embeddings, words_data)

#Sen-Level Bi-RNN Layers
word_embedded = tf.reshape(word_embedded, [-1, SEN_LENTH, EMBEDDING_SIZE])
with tf.variable_scope("word_encoder"):
    gru_fw = GRUCell(HIDDEN_SIZE)
    gru_bw = GRUCell(HIDDEN_SIZE)
    rnn_outputs, _ = bi_rnn(cell_fw=gru_fw,
                        cell_bw=gru_bw,
                        inputs=word_embedded,
                        sequence_length=length(word_embedded),
                        dtype=tf.float32)

#Attention Layer
attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)

#Docu-Level Bi-RNN Layers
attention_output = tf.reshape(attention_output, [-1, DOCU_LENTH, HIDDEN_SIZE * 2])
with tf.variable_scope("sent_encoder"):
    gru_fw2 = GRUCell(HIDDEN_SIZE)
    gru_bw2= GRUCell(HIDDEN_SIZE)
    sen_rnn_outputs, _ = bi_rnn(cell_fw=gru_fw2,
                            cell_bw=gru_bw2,
                            inputs=attention_output,
                            sequence_length=length(attention_output),
                            dtype=tf.float32)

#Attention Layer
docu_atten_output, alphas = attention(sen_rnn_outputs, ATTENTION_SIZE, return_alphas=True)

# #Dropout
# drop_out = tf.nn.dropout(docu_atten_output, keep_prob_ph)

#Full Connected
W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))
b = tf.Variable(tf.constant(0., shape=[1]))
out = tf.nn.xw_plus_b(docu_atten_output, W, b)
out = tf.squeeze(out)

#Loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=out))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=loss)

# Accuracy metric
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(out)), labels), tf.float32))

train_batch_generator = batch_generator(train_X, train_Y, BATCH_SIZE)
test_batch_generator = batch_generator(test_X, test_Y, BATCH_SIZE)

saver = tf.train.Saver()
with tf.Session() as sess:
    # saver.restore(sess, "/Users/zrzn/Downloads/imda_classify/embedding.ckpt")
    sess.run(tf.global_variables_initializer())
    print("Start learning...")
    for epoch in range(NUM_EPOCHS):
        loss_train = 0
        loss_test = 0
        accuracy_train = 0
        accuracy_test = 0

        print("epoch: {}\t".format(epoch), end="")

        # Training
        num_batches = train_X.shape[0] // BATCH_SIZE
        for b in range(num_batches):
            x_batch, y_batch = next(train_batch_generator)
            seq_len = np.array([40 for x in range(BATCH_SIZE * SEN_LENTH)])  # actual lengths of sequences
            docu_len = np.array([10 for x in range(BATCH_SIZE)])
            loss_tr, acc, _ = sess.run([loss, accuracy, optimizer],
                                       feed_dict={words_data: x_batch,
                                                  labels: y_batch,
                                                  keep_prob_ph: KEEP_PROB})
            accuracy_train += acc
            loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
        accuracy_train /= num_batches

        # Testing
        num_batches = test_X.shape[0] // BATCH_SIZE
        for b in range(num_batches):
            x_batch, y_batch = next(test_batch_generator)
            seq_len = np.array([40 for x in range(BATCH_SIZE * DOCU_LENTH)])  # actual lengths of sequences
            docu_len = np.array([10 for x in range(BATCH_SIZE)])
            loss_test_batch, acc = sess.run([loss, accuracy],
                                            feed_dict={words_data: x_batch,
                                                       labels: y_batch,
                                                       keep_prob_ph: 1.0})
            accuracy_test += acc
            loss_test += loss_test_batch
        accuracy_test /= num_batches
        loss_test /= num_batches
        print("accuracy == ", accuracy_test)
        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))
