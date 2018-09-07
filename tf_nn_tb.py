from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def main():
  # Import data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784], name = 'x')
  y = tf.placeholder(tf.float32, [None, 10])

  #First layer
  W1 = tf.Variable(tf.truncated_normal([784, 20]), name = 'W1')
  b1 = tf.Variable(tf.truncated_normal([20]), name = 'b1')
  z1 = tf.matmul(x, W1) + b1
  h = tf.nn.sigmoid(z1)
  #dimension is now [None, 20]

  #Second layer
  W2 = tf.Variable(tf.truncated_normal([20, 10]), name = 'W2')
  b2 = tf.Variable(tf.truncated_normal([10]))
  y_ = tf.matmul(h, W2) + b2

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  tf.summary.histogram('W2', W2)
  tf.summary.scalar('cross_entropy_loss', loss)

  #Initialize session
  sess = tf.Session()

  
  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('./summaries', sess.graph)


  sess.run(tf.global_variables_initializer())


  
  for step in range(5000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      summary, _ = sess.run([summary_op, train_step], feed_dict = {x: batch_xs, y: batch_ys})
      summary_writer.add_summary(summary, step)
  
  correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))
  print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))


if __name__ == '__main__':
    main()