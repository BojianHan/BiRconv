import numpy as np
import theano
import theano.tensor as T
import math
import itertools
import matplotlib.pyplot as plt
import random

X = T.dvector('X')
W1 = T.dmatrix('W1')
b1 = T.dvector('b1')
W2 = T.dmatrix('W2')
b2 = T.dvector('b2')
W3 = T.dmatrix('W3')
b3 = T.dvector('b3')

target = T.dvector('target')

h1 = T.nnet.relu(T.dot(X, W1) + b1)
#h2 = T.nnet.relu(T.dot(h1, W2) + b2)
#output = T.nnet.softmax(T.dot(h2, W3) + b3)
output = T.nnet.softmax(T.dot(h1, W2) + b2)

# cost = ((output - target) ** 2).mean()
cost = T.nnet.binary_crossentropy(output, target).mean()

#[gw1, gb1, gw2, gb2, gw3, gb3] = T.grad(cost, [W1, b1, W2, b2, W3, b3])
[gw1, gb1, gw2, gb2] = T.grad(cost, [W1, b1, W2, b2])
#fn = theano.function([X, W1, b1, W2, b2, W3, b3, target], [output, gw1, gb1, gw2, gb2, gw3, gb3])
fn = theano.function([X, W1, b1, W2, b2, target], [output, gw1, gb1, gw2, gb2, h1])

num_input = 50
num_output = 125
compare_limit = 4
num_input_sq = num_input * num_input
learning_rate = 1
epoch = 200
batch_size = 500
num_data = epoch * batch_size

np_W1 = np.random.uniform(size=(num_input, num_input_sq), low=-.1, high=.1)
np_W2 = np.random.uniform(size=(num_input_sq, num_output), low=-0.1, high=0.1)
np_b1 = np.random.uniform(size=(num_input_sq), low=-0.1, high=0.1)
np_b2 = np.random.uniform(size=(num_output), low=-0.1, high=0.1)

#np_W3 = np.random.uniform(size=(num_input_sq, num_output), low=-0.1, high=0.1)
#np_b3 = np.random.uniform(size=(num_output), low=-0.1, high=0.1)

print "Num_Parens:", sum([np.prod(weight.shape) for weight in [np_W1, np_W2, np_b1, np_b2]])

selector = []
for i in range(num_output):
    selector.append(random.sample(range(num_input),compare_limit))
input_data = np.random.uniform(size=(num_data, num_input), low=-1, high=1)
target_data = np.zeros((num_data, num_output))
for i in range(num_output):
    target_data[:,i] = 1 * ((np.diff(input_data[:,selector[i]]) > 0).sum(axis=1) == compare_limit - 1)

count_list = []
for batch_index in range(epoch):
    count = 0
    #weights = [np_W1, np_b1, np_W2, np_b2, np_W3, np_b3]
    weights = [np_W1, np_b1, np_W2, np_b2]
    update_grads = [np.zeros(weight.shape) for weight in weights]
    for i in range(batch_size):
        data_index = batch_index * batch_size + i
        np_I = input_data[data_index]
        np_T = target_data[data_index]

        #np_O, gw1, gb1, gw2, gb2, gw3, gb3 = fn(np_I, np_W1, np_b1, np_W2, np_b2, np_W3, np_b3, np_T)
        np_O, gw1, gb1, gw2, gb2, np_h2 = fn(np_I, np_W1, np_b1, np_W2, np_b2, np_T)
        #grads = [gw1, gb1, gw2, gb2, gw3, gb3]
        grads = [gw1, gb1, gw2, gb2]
        for j in range(len(grads)):
            update_grads[j] += grads[j]
        if np_T[np.argmax(np_O)] == 1 or np_T.sum() < 1:
            count += 1
    for k in range(len(weights)):
        weights[k] -= learning_rate * update_grads[k]

    print batch_index, 100.0 * count / batch_size
    count_list.append(100.0 * count / batch_size)
    learning_rate *= 0.999

print sum(count_list) / epoch
plt.plot(range(epoch), count_list)
plt.show()
