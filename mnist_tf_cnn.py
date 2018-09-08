from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def conv_model(x):
  x = tf.reshape(x, [tf.shape(x)[0], 28, 28, 1])
  x = tf.layers.conv2d(x, 32, [5, 5], activation = 'relu')
  x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)
  x = tf.layers.conv2d(x, 64, [5, 5], activation = 'relu')
  x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)
  x = tf.layers.flatten(x)
  x = tf.layers.dense(x, 1000, activation = 'relu')
  x = tf.layers.dense(x, 10, activation = 'softmax')
  return x

def main():
  # Import data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y = conv_model(x)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  
  summary_writer = tf.summary.FileWriter('./summaries_cnn', sess.graph)
  tf.global_variables_initializer().run()
#   # Train
#   for _ in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#   # Test trained model
#   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#   print(sess.run(accuracy, feed_dict={x: mnist.test.images,
# y_: mnist.test.labels}))
  

if __name__ == '__main__':
  main()