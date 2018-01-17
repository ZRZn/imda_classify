#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.datasets import imdb
import tensorflow as tf
import numpy as np
import math



skip = 2
sen_index = 0
word_index = skip
vocabulary_size = 30000

(X_train, y_train), (X_test, y_test) = imdb.load_data(path="/Users/ZRZn1/Downloads/imdb.npz", num_words=30000,
                                                      index_from=3)
dictionary = imdb.get_word_index(path="/Users/ZRZn1/Downloads/imdb_word_index.json")
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

def generate_batch(X, batch_size, num_skips, skip_window):
    global sen_index
    global word_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # print("最初的word_index == " + str(word_index))
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    i = 0
    while i < batch_size:
        while word_index < len(X[sen_index]) - skip_window and i < batch_size:
            for a in range(0, skip_window):
                batch[i] = X[sen_index][word_index]
                labels[i, 0] = X[sen_index][word_index + a - 2]
                i += 1
            for a in range(0, skip_window):
                batch[i] = X[sen_index][word_index]
                labels[i, 0] = X[sen_index][word_index + a + 1]
                i += 1
            word_index += 1
        if word_index == len(X[sen_index]) - skip_window:
            sen_index = (sen_index + 1) % len(X)
            word_index = skip_window
    return batch, labels






batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                     num_sampled=num_sampled, num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

saver = tf.train.Saver({'embeddings': embeddings})












# Step 5: Begin training.
num_steps = 50001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in range(num_steps):
    # if num_steps <= 50000:
    #     batch_inputs, batch_labels = generate_batch(
    #         X_train, batch_size, num_skips, skip_window)
    # elif num_steps == 50001:
    #     word_index = skip
    #     sen_index = 0
    #     batch_inputs, batch_labels = generate_batch(
    #         X_test, batch_size, num_skips, skip_window)
    # else:
    #     batch_inputs, batch_labels = generate_batch(
    #         X_test, batch_size, num_skips, skip_window)

    batch_inputs, batch_labels = generate_batch(
        X_train, batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
  save_path = saver.save(session, "/Users/ZRZn1/Downloads/embedding.ckpt")
  print("save_path == " + str(save_path))