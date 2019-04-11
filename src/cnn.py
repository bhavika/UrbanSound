import tensorflow as tf
import numpy as np
from src.create_rep import *
import matplotlib.pyplot as plt
from src.constants import *


tr_features = np.load(train_features_pickle)
tr_labels = np.load(train_labels_pickle)
tr_labels = one_hot_encode(tr_labels)

ts_features = np.load(test_features_pickle)
ts_labels = np.load(test_labels_pickle)
ts_labels = one_hot_encode(ts_labels)


def apply_convolution(x, cnn_kernel_size, num_channels, depth):
    weights = tf.Variable(tf.truncated_normal(shape=[cnn_kernel_size, cnn_kernel_size, num_channels, depth], stddev=0.1))
    biases = tf.Variable(tf.constant(1.0, shape=([depth])))
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME'), biases))


X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

cov = apply_convolution(X, cnn_kernel_size, num_channels, cnn_depth)

shape = cov.get_shape().as_list()

cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = tf.Variable(tf.truncated_normal(shape=[shape[1] * shape[2] * cnn_depth, cnn_num_hidden]))
f_biases = tf.Variable(tf.constant(1.0, shape=[cnn_num_hidden]))
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights), f_biases))

out_weights = tf.Variable(tf.truncated_normal(shape=[cnn_num_hidden, num_labels]))
out_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=cnn_learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost_history = np.empty(shape=[1], dtype=float)


with tf.Session() as session:
    tf.global_variables_initializer().run()

    for itr in range(cnn_total_iterations):
        offset = (itr * cnn_batch_size) % (tr_labels.shape[0] - cnn_batch_size)
        batch_x = tr_features[offset:(offset + cnn_batch_size), :, :, :]
        batch_y = tr_labels[offset:(offset + cnn_batch_size)]

        _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        cost_history = np.append(cost_history, c)

    print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(cost_history)
    plt.axis([0, cnn_total_iterations, 0, np.max(cost_history)])
    plt.show()

    fw = tf.summary.FileWriter('../output/cnn', graph=session.graph)
    fw.close()
