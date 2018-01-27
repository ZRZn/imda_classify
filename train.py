#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
import os
from attention import attention
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.layers import fully_connected
import numpy as np
from utils import *
from path import *

NUM_EPOCHS = 100
BATCH_SIZE = 64
HIDDEN_SIZE = 50
USR_SIZE = 1310
PRD_SIZE = 1635
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 100
KEEP_PROB = 0.8
DELTA = 0.5
PosTrain = False

#Load Data
train_fir = open(all_path + "train_out_np.pkl", "rb")
test_fir = open(all_path + "test_out_np.pkl", "rb")
pos_train_x = pickle.load(train_fir)
pos_train_y = pickle.load(train_fir)
pos_train_u = pickle.load(train_fir)
pos_train_p = pickle.load(train_fir)
neg_train_x = pickle.load(train_fir)
neg_train_y = pickle.load(train_fir)
neg_train_u = pickle.load(train_fir)
neg_train_p = pickle.load(train_fir)


pos_test_x = pickle.load(test_fir)
pos_test_y = pickle.load(test_fir)
pos_test_u = pickle.load(test_fir)
pos_test_p = pickle.load(test_fir)
neg_test_x = pickle.load(test_fir)
neg_test_y = pickle.load(test_fir)
neg_test_u = pickle.load(test_fir)
neg_test_p = pickle.load(test_fir)

train_fir.close()
test_fir.close()


if PosTrain:
    train_X = pos_train_x
    train_Y = pos_train_y
    train_U = pos_train_u
    train_P = pos_train_p
    test_X = pos_test_x
    test_Y = pos_test_y
    test_U = pos_test_u
    test_P = pos_test_p
else:
    train_X = neg_train_x
    train_Y = neg_train_y
    train_U = neg_train_u
    train_P = neg_train_p
    test_X = neg_test_x
    test_Y = neg_test_y
    test_U = neg_test_u
    test_P = neg_test_p

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


#placeholders
words_data = tf.placeholder(tf.int32, [BATCH_SIZE, None, None])
labels = tf.placeholder(tf.float32, [BATCH_SIZE, 10])
user_ph = tf.placeholder(tf.int32, [BATCH_SIZE])
prd_ph = tf.placeholder(tf.int32, [BATCH_SIZE])
keep_prob_ph = tf.placeholder(tf.float32)
sen_len_ph = tf.placeholder(tf.int32)
sen_num_ph = tf.placeholder(tf.int32)

#Embedding Layer
emb_fir = open(all_path + "emb_array.pkl", "rb")
emb_np = pickle.load(emb_fir)
emb_fir.close()
embeddings = tf.Variable(initial_value=emb_np, trainable=True, name="embeddings")
word_embedded = tf.nn.embedding_lookup(embeddings, words_data)

#User&Prd embedding
usr_emb = tf.Variable(tf.zeros(shape=[USR_SIZE, HIDDEN_SIZE * 2]), trainable=True, dtype=tf.float32)
prd_emb = tf.Variable(tf.zeros(shape=[PRD_SIZE, HIDDEN_SIZE * 2]), trainable=True, dtype=tf.float32)
#shape = [B, EBD_SIZE]
usr_data = tf.nn.embedding_lookup(usr_emb, user_ph)
prd_data = tf.nn.embedding_lookup(prd_emb, prd_ph)



#Sen-Level Bi-RNN Layers
word_embedded = tf.reshape(word_embedded, [-1, sen_len_ph, EMBEDDING_SIZE])
usr_word = tf.reshape(usr_data, (BATCH_SIZE, -1, HIDDEN_SIZE * 2))
usr_word = tf.tile(usr_word, multiples=(sen_num_ph, sen_len_ph, 1))
prd_word = tf.reshape(prd_data, (BATCH_SIZE, -1, HIDDEN_SIZE * 2))
prd_word = tf.tile(prd_word, multiples=(sen_num_ph, sen_len_ph, 1))
with tf.variable_scope("word_encoder"):
    gru_fw = GRUCell(HIDDEN_SIZE)
    gru_bw = GRUCell(HIDDEN_SIZE)
    (f_out, b_out), _ = bi_rnn(cell_fw=gru_fw,
                        cell_bw=gru_bw,
                        inputs=word_embedded,
                        sequence_length=length(word_embedded),
                        dtype=tf.float32)
    rnn_outputs = tf.concat((f_out, b_out), axis=2)

#Attention Layer
attention_output = attention(rnn_outputs, ATTENTION_SIZE, usr_word, prd_word)

#Docu-Level Bi-RNN Layers
attention_output = tf.reshape(attention_output, [-1, sen_num_ph, HIDDEN_SIZE * 2])
usr_sen = tf.reshape(usr_data, (BATCH_SIZE, -1, HIDDEN_SIZE * 2))
usr_sen = tf.tile(usr_sen, multiples=(1, sen_num_ph, 1))
prd_sen = tf.reshape(prd_data, (BATCH_SIZE, -1, HIDDEN_SIZE * 2))
prd_sen = tf.tile(prd_sen, multiples=(1, sen_num_ph, 1))
with tf.variable_scope("sent_encoder"):
    gru_fw2 = GRUCell(HIDDEN_SIZE)
    gru_bw2 = GRUCell(HIDDEN_SIZE)
    (f_out2, b_out2), _ = bi_rnn(cell_fw=gru_fw2,
                            cell_bw=gru_bw2,
                            inputs=attention_output,
                            sequence_length=length(attention_output),
                            dtype=tf.float32)
    sen_rnn_outputs = tf.concat((f_out2, b_out2), axis=2)

#Attention Layer
docu_atten_output = attention(sen_rnn_outputs, ATTENTION_SIZE, usr_sen, prd_sen)

#Dropout
drop_out = tf.nn.dropout(docu_atten_output, keep_prob_ph)

#Full Connected
W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 10], stddev=0.1))
b = tf.Variable(tf.constant(0., shape=[10]))
out = tf.nn.xw_plus_b(drop_out, W, b)
out = tf.squeeze(out)

#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out))
optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss=loss)


# Accuracy metric
predict = tf.argmax(out, axis=1, name='predict')
label = tf.argmax(labels, axis=1, name='label')
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Start learning...")

    for epoch in range(NUM_EPOCHS):
        loss_train = 0
        loss_test = 0
        accuracy_train = 0
        accuracy_test = 0

        print("epoch: {}\t".format(epoch), end="")


        # Training
        num_batches = len(train_X) // BATCH_SIZE

        indices = np.arange(num_batches)
        np.random.shuffle(indices)

        for b in range(num_batches):
            count = indices[b]
            print("迭代轮数:", count)
            x_train = train_X[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
            y_train = train_Y[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
            u_train = train_U[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
            p_train = train_P[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
            sen_len = len(x_train[0][0])
            sen_num = len(x_train[0])
            # seq_len = np.array([40 for x in range(BATCH_SIZE * SEN_LENTH)])  # actual lengths of sequences
            # docu_len = np.array([10 for x in range(BATCH_SIZE)])
            loss_tr, acc, _ = sess.run([loss, accuracy, optimizer],
                                       feed_dict={words_data: x_train,
                                                  labels: y_train,
                                                  user_ph: u_train,
                                                  prd_ph: p_train,
                                                  sen_len_ph: sen_len,
                                                  sen_num_ph: sen_num,
                                                  keep_prob_ph: KEEP_PROB})
            accuracy_train += acc
            loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
            if b % 50 == 0 and b > 100:
                print("accuracy_train" == accuracy_train / (b + 1))
                # Testing
                test_batches = len(test_X) // BATCH_SIZE
                for z in range(test_batches):
                    x_test = test_X[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                    y_test = test_Y[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                    u_test = test_U[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                    p_test = test_P[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                    test_len = len(x_test[0][0])
                    test_num = len(x_test[0])
                    # seq_len = np.array([40 for x in range(BATCH_SIZE * DOCU_LENTH)])  # actual lengths of sequences
                    # docu_len = np.array([10 for x in range(BATCH_SIZE)])
                    loss_test_batch, test_acc = sess.run([loss, accuracy],
                                                    feed_dict={words_data: x_test,
                                                               labels: y_test,
                                                               user_ph: u_test,
                                                               prd_ph: p_test,
                                                               sen_len_ph: test_len,
                                                               sen_num_ph: test_num,
                                                               keep_prob_ph: 1.0})
                    accuracy_test += test_acc
                    loss_test += loss_test_batch
                accuracy_test /= test_batches
                loss_test /= test_batches
                print("accuracy_test == ", accuracy_test)
        accuracy_train /= num_batches

        # # Testing
        # num_batches = test_X.shape[0] // BATCH_SIZE
        # for b in range(num_batches):
        #     x_batch, y_batch = next(test_batch_generator)
        #     # seq_len = np.array([40 for x in range(BATCH_SIZE * DOCU_LENTH)])  # actual lengths of sequences
        #     # docu_len = np.array([10 for x in range(BATCH_SIZE)])
        #     loss_test_batch, acc = sess.run([loss, accuracy],
        #                                     feed_dict={words_data: x_batch,
        #                                                labels: y_batch,
        #                                                keep_prob_ph: 1.0})
        #     accuracy_test += acc
        #     loss_test += loss_test_batch
        # accuracy_test /= num_batches
        # loss_test /= num_batches
        # print("accuracy == ", accuracy_test)
        # print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
        #     loss_train, loss_test, accuracy_train, accuracy_test
        # ))
