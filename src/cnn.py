import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from src.explore import get_data
from src.config import *


def apply_convolution(x, cnn_kernel_size, num_channels, depth):
    weights = tf.Variable(tf.truncated_normal(shape=[cnn_kernel_size, cnn_kernel_size, num_channels, depth], stddev=0.1))
    biases = tf.Variable(tf.constant(1.0, shape=([depth])))
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME'), biases))


def cnn():
    """
    A simple 2-layer convolutional neural network implemented in Tensorflow.

    :return:
    """
    X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    cov = apply_convolution(X, cnn_kernel_size, num_channels, cnn_depth)

    shape = cov.get_shape().as_list()

    cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

    f_weights = tf.Variable(tf.truncated_normal(shape=[shape[1] * shape[2] * cnn_depth, cnn_num_hidden]))
    f_biases = tf.Variable(tf.constant(1.0, shape=[cnn_num_hidden]))
    f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights), f_biases))

    # placeholder for weights output
    out_weights = tf.Variable(tf.truncated_normal(shape=[cnn_num_hidden, num_labels]))
    out_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # output prediction
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

    loss = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=cnn_learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost_history = np.empty(shape=[1], dtype=float)

    # get train and test data
    train_arr, train_labels_arr, test_arr, test_labels_arr = get_data()

    # start Tensorflow training session

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        for itr in range(cnn_total_iterations):
            offset = (itr * cnn_batch_size) % (train_labels_arr.shape[0] - cnn_batch_size)
            batch_x = train_arr[offset:(offset + cnn_batch_size), :, :, :]
            batch_y = train_labels_arr[offset:(offset + cnn_batch_size)]

            _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)

        print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: test_arr, Y: test_labels_arr}), 3))

        fig = plt.figure(figsize=(15, 10))
        plt.plot(cost_history)
        plt.axis([0, cnn_total_iterations, 0, np.max(cost_history)])
        plt.show()

        fw = tf.summary.FileWriter(os.path.join(os.getenv('OUTPUT'), 'cnn'), graph=session.graph)
        fw.close()


if __name__ == '__main__':
    cnn()
