# BiRconv 1 layer
# TensorFlow implementation of multilayer biRconv network

import tensorflow as tf
import numpy as np

BATCH_SIZE = 100
EPOCHS = 100
#INPUT_SIZE = 3
#OUTPUT_SIZE = 3

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

        # Expand weights by batch size (implicitly done by broadcasting)
        #W1_e = tf.reshape(W1, [-1, input_size, input_size])
        #W1_er = tf.tile(W1_e, [batch_size, 1, 1])
        #W2_e = tf.reshape(W2, [-1, input_size, input_size])
        #W2_er = tf.tile(W2_e, [batch_size, 1, 1])
        #b_hid_e = tf.reshape(b_hid, [-1, input_size, input_size])
        #b_hid_er = tf.tile(b_hid_e, [batch_size, 1, 1])

        # biRconv
        H = tf.nn.relu(X_er * W1 + X_ert * W2 + b_hid)

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

    return (output_vector, H, W1, W2, b_hid, W_mlp, W_hid, b_out)

def load_data():
    X_train = np.genfromtxt('X_train.csv', delimiter=',')
    Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
    X_test = np.genfromtxt('X_test.csv', delimiter=',')
    Y_test = np.genfromtxt('Y_test.csv', delimiter=',')

    return (X_train, Y_train, X_test, Y_test)

def main():
    X_train, Y_train, X_test, Y_test = load_data()

    # Unfortunately due to implementation details, BATCH_SIZE is fixed at start
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, X_train.shape[1]])
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE, Y_train.shape[1]])

    L1, H1,W11,W21,b_hid1,W_mlp1,W_hid1,b_out1 = biRconv_layer('l1', x, Y_train.shape[1])
    y_pred = L1
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))
    #top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_pred, tf.argmax(y, 1), 5), tf.float32))
    accs = [tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_pred, tf.argmax(y, 1), i), tf.float32)) for i in range(1, 6)]

    X_vis = np.arange(X_train.shape[1]) / 50.0 - 0.99
    X_vis = X_vis.reshape((1, X_train.shape[1])).repeat(BATCH_SIZE, axis=0)
    Y_vis = np.zeros((BATCH_SIZE, Y_train.shape[1]))

    # NOTE: instances should be multiple of BATCH_SIZE
    train_instances = np.shape(X_train)[0]
    test_instances = np.shape(X_test)[0]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        acts = np.zeros((EPOCHS+1, X_train.shape[1], X_train.shape[1]))
        acts[0, :, :] = sess.run(H1, feed_dict={x:X_vis, y:Y_vis})[0,:,:]

        for e in xrange(EPOCHS):
            for batch in xrange(train_instances / BATCH_SIZE):
                X_batch = X_train[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE, :]
                Y_batch = Y_train[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE, :]

                _, loss_value = sess.run([train_step, loss], feed_dict={x:X_batch, y:Y_batch})
                print 'Iterations: %d, Loss: %.5f' % (e * train_instances + (batch+1)*BATCH_SIZE, loss_value)

            # Evaluate test accuracy at end of each epoch
            #acc_sum = 0.0
            #top5_sum = 0.0
            accs_sum = np.zeros((5))
            for batch in xrange(test_instances / BATCH_SIZE):
                X_batch = X_test[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE, :]
                Y_batch = Y_test[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE, :]
                #[a, b] = sess.run([accuracy, top5], feed_dict={x:X_batch, y:Y_batch})
                #acc_sum += a
                #top5_sum += b
                acc_vals = sess.run(accs, feed_dict={x:X_batch, y:Y_batch})
                for i, a in enumerate(acc_vals):
                    accs_sum[i] += a
            accs_sum /= (test_instances / BATCH_SIZE)
            print '>>> Epoch %d, Iterations: %d, Top-1: %.5f, Top-2: %.5f, Top-3: %.5f, Top-4: %.5f, Top-5: %.5f' % (e+1, (e+1)*train_instances, accs_sum[0], accs_sum[1], accs_sum[2], accs_sum[3], accs_sum[4])
            #print '>>> Epoch %d, Iterations: %d, Top5: %.5f, Test Accuracy: %.5f' % (e+1, (e+1) * train_instances, top5_sum / (test_instances / BATCH_SIZE), acc_sum / (test_instances / BATCH_SIZE))

            acts[e+1, :, :] = sess.run(H1, feed_dict={x:X_vis, y:Y_vis})[0,:,:]

        #print y_pred.eval(feed_dict={x:[[1,2,3],[2,3,4]], y:[[1,2,3],[2,3,4]]})
        #train_step.run(feed_dict={x:[[1,2,3],[2,3,4]], y:[[1,2,3],[2,3,4]]})
        #print W11.eval()

        #merged = tf.merge_all_summaries()
        #writer = tf.train.SummaryWriter('logs', sess.graph_def)
    np.save('acts.npy', acts)

if __name__ == '__main__':
    main()
