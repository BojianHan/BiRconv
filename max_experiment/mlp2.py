# MLP 1 hidden layer + skip connection

import tensorflow as tf
import numpy as np

BATCH_SIZE = 100
EPOCHS = 100
HIDDEN = 100

def initialize_weight(name, shape):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

def initialize_bias(name, shape, value):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))

def load_data():
    X_train = np.genfromtxt('X_train.csv', delimiter=',')
    Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
    X_test = np.genfromtxt('X_test.csv', delimiter=',')
    Y_test = np.genfromtxt('Y_test.csv', delimiter=',')

    return (X_train, Y_train, X_test, Y_test)

def main():
    X_train, Y_train, X_test, Y_test = load_data()

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
    y = tf.placeholder(tf.int64, shape=[None, Y_train.shape[1]])

    W1 = initialize_weight('W1', (X_train.shape[1], HIDDEN))
    b1 = initialize_bias('b1', (1, HIDDEN), 0.1)
    h = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = initialize_weight('W2', (HIDDEN, Y_train.shape[1]))
    b2 = initialize_bias('b2', (1, Y_train.shape[1]), 0.1)

    W3 = initialize_weight('W3', (X_train.shape[1], Y_train.shape[1]))

    y_pred = tf.nn.relu(tf.matmul(h, W2) + tf.matmul(x, W3) + b2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))
    top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_pred, tf.argmax(y, 1), 5), tf.float32))

    # NOTE: instances should be multiple of BATCH_SIZE
    train_instances = np.shape(X_train)[0]
    test_instances = np.shape(X_test)[0]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for e in xrange(EPOCHS):
            for batch in xrange(train_instances / BATCH_SIZE):
                X_batch = X_train[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE, :]
                Y_batch = Y_train[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE, :]

                _, loss_value = sess.run([train_step, loss], feed_dict={x:X_batch, y:Y_batch})
                print 'Iterations: %d, Loss: %.5f' % (e * train_instances + (batch+1)*BATCH_SIZE, loss_value)

            # Evaluate test accuracy at end of each epoch
            test_acc = sess.run(accuracy, feed_dict={x:X_test, y:Y_test})
            top5_sum += sess.run(top5, feed_dict={x:X_batch, y:Y_batch})
            print '>>> Epoch %d, Iterations: %d, Top5: %.5f, Test Accuracy: %.5f' % (e+1, (e+1) * train_instances, top5_sum, test_acc)

if __name__ == '__main__':
    main()
