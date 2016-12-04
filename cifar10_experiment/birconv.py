
# TensorFlow implementation of multilayer biRconv network

import tensorflow as tf
import numpy as np

BATCH_SIZE = 2
INPUT_SIZE = 3
OUTPUT_SIZE = 3

def initialize_weight(name, shape):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

def initialize_bias(name, shape, value):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))

def biRconv_layer(name, input_vector, output_size):
    batch_size = int(input_vector.get_shape()[0])
    input_size = int(input_vector.get_shape()[1])

    with tf.variable_scope(name) as scope:
        # Initialize bi-conv weights (debug)
        #W1 = tf.Variable([[1]*input_size]*input_size, dtype=tf.float32)
        #W2 = tf.Variable([[1]*input_size]*input_size, dtype=tf.float32)
        #b_hid = tf.constant(0, shape=[input_size, input_size], dtype=tf.float32)

        # Initialize bi-conv weights
        W1 = initialize_weight('W1', [input_size, input_size])
        W2 = initialize_weight('W2', [input_size, input_size])
        b_hid = initialize_bias('b_hid', [input_size, input_size], 0.1)

        # Expand and repeat input vectors
        X_e = tf.reshape(input_vector, [batch_size, input_size, -1])
        X_er = tf.tile(X_e, [1, 1, input_size])
        X_ert = tf.transpose(X_er, [0,2,1])

        # Expand weights by batch size
        W1_e = tf.reshape(W1, [-1, input_size, input_size])
        W1_er = tf.tile(W1_e, [batch_size, 1, 1])
        W2_e = tf.reshape(W2, [-1, input_size, input_size])
        W2_er = tf.tile(W2_e, [batch_size, 1, 1])
        b_hid_e = tf.reshape(b_hid, [-1, input_size, input_size])
        b_hid_er = tf.tile(b_hid_e, [batch_size, 1, 1])

        # biRconv
        H = tf.nn.relu(X_er * W1_er + X_ert * W2_er + b_hid_er)

        # FC initalization (debug)
        #W_mlp = tf.Variable([[1]*output_size]*input_size, dtype=tf.float32)
        #W_hid = tf.Variable([[1]*output_size]*(input_size**2), dtype=tf.float32)
        #b_out = tf.constant(0, shape=[1, output_size], dtype=tf.float32)

        # FC initalization
        W_mlp = initialize_weight('W_mlp', [input_size, output_size])
        W_hid = initialize_weight('W_hid', [input_size**2, output_size])
        b_out = initialize_bias('b_out', [1, output_size], 0.1)

        # FC
        H_f = tf.reshape(H, [-1, input_size**2])
        output_vector = tf.nn.relu(tf.matmul(input_vector, W_mlp) + tf.matmul(H_f, W_hid) + b_out)

    return (output_vector, W1, W2, b_hid, W_mlp, W_hid, b_out)


def main():
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_SIZE])
    y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_SIZE])

    L1, W11,W21,b_hid1,W_mlp1,W_hid1,b_out1 = biRconv_layer('l1', x, 4)
    L2, W12,W22,b_hid2,W_mlp2,W_hid2,b_out2 = biRconv_layer('l2', L1, OUTPUT_SIZE)
    y_pred = L2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print y_pred.eval(feed_dict={x:[[1,2,3],[2,3,4]], y:[[1,2,3],[2,3,4]]})
        train_step.run(feed_dict={x:[[1,2,3],[2,3,4]], y:[[1,2,3],[2,3,4]]})
        print W11.eval()

        #merged = tf.merge_all_summaries()
        #writer = tf.train.SummaryWriter('logs', sess.graph_def)

if __name__ == '__main__':
    main()
