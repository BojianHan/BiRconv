# Basic CNN for part 1

import tensorflow as tf

ITERS = 1
BATCH = 100

# Load input data (TensorFlow happens to have some code that does this)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def initialize_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def initialize_bias(shape):
    # Initialize positive to prevent dead neurons
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])
img_x = tf.reshape(x, [-1,28,28,1])

# Conv1
conv1_W = initialize_weight([5,5,1,32])
conv1_b = initialize_bias([32])

conv1_a = tf.nn.conv2d(img_x, conv1_W, [1,1,1,1], padding='VALID') + conv1_b
conv1_a = tf.nn.relu(conv1_a)
conv1_a = tf.pad(conv1_a, [[0,0],[1,1],[1,1],[0,0]], 'CONSTANT')
conv1_a = tf.nn.max_pool(conv1_a, ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')

# Conv2
conv2_W = initialize_weight([5,5,32,64])
conv2_b = initialize_bias([64])

conv2_a = tf.nn.conv2d(conv1_a, conv2_W, [1,1,1,1], padding='VALID') + conv2_b
conv2_a = tf.nn.relu(conv2_a)
conv2_a = tf.pad(conv2_a, [[0,0],[1,1],[1,1],[0,0]], 'CONSTANT')
conv2_a = tf.nn.max_pool(conv2_a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
conv2_a = tf.reshape(conv2_a, [-1, 3*3*64])

# FC3
fc3_W = initialize_weight([3*3*64, 200])
fc3_b = initialize_bias([200])

fc3_a = tf.matmul(conv2_a, fc3_W) + fc3_b
fc3_a = tf.nn.relu(fc3_a)

# FC4
fc4_W = initialize_weight([200, 10])
fc4_b = initialize_bias([10])

fc4_a = tf.matmul(fc3_a, fc4_W) + fc4_b
pred = tf.argmax(fc4_a, 1)
true = tf.argmax(y, 1)

# Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc4_a, y))
trainer = tf.train.AdamOptimizer().minimize(loss)
correct = tf.equal(pred, true)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

# Training
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(ITERS):
        batch = mnist.train.next_batch(BATCH)
        if i % 100:
            train_acc = accuracy.eval(
                    feed_dict={x:batch[0], y:batch[1]})
            test_acc = accuracy.eval(
                    feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print "%d %g %g" % (i * BATCH, train_acc, test_acc)

        trainer.run(
                feed_dict={x:batch[0], y:batch[1]})

    test_acc = accuracy.eval(
            feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print "Final test accuracy: %g" % test_acc

    test_pred = pred.eval(
            feed_dict={x:mnist.test.images, y:mnist.test.labels})
    true_pred = true.eval(
            feed_dict={x:mnist.test.images, y:mnist.test.labels})

    summary_writer = tf.train.SummaryWriter('outputs', sess.graph)
    num_added = 0
    for i in range(test_pred.shape[0]):
        if test_pred[i] != true_pred[i]:
            img = tf.reshape(mnist.test.images[i, :], [-1, 28, 28, 1])
            img_summ = tf.image_summary('img%d_pred%d_true%d' % (i, test_pred[i], true_pred[i]), img, max_images=1)
            summary_writer.add_summary(sess.run(img_summ))
            num_added += 1
            if num_added == 10:
                break

    #conv1_summ = tf.image_summary('conv1', tf.transpose(conv1_W, [3,0,1,2]), max_images=32)
    conv1_summ = tf.image_summary('conv1', put_kernels_on_grid(conv1_W, 4, 8), max_images=1)
    summary_writer.add_summary(sess.run(conv1_summ))

