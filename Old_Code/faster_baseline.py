import numpy as np
import theano
import theano.tensor as T
import math
import itertools
import matplotlib.pyplot as plt
import random

X = T.dmatrix('X')
W1 = T.dmatrix('W1')
b1 = T.dvector('b1')
W2 = T.dmatrix('W2')
b2 = T.dvector('b2')
target = T.dmatrix('target')

bs = T.shape(X)[0]
h1 = T.nnet.relu(T.dot(X, W1) + T.tile(b1, (bs,1)))
output = T.nnet.softmax(T.dot(h1, W2) + T.tile(b2, (bs,1)))

# cost = ((output - target) ** 2).mean()
cost = T.nnet.binary_crossentropy(output, target).mean(axis=1).sum()

[gw1, gb1, gw2, gb2] = T.grad(cost, [W1, b1, W2, b2])
fn = theano.function([X, W1, b1, W2, b2, target], [output, gw1, gb1, gw2, gb2, h1])

num_input = 50
num_output = 125
compare_limit = 5
learning_rate = 1
epoch = 1000
batch_size = 500
num_data = epoch * batch_size
num_input_sq = num_input * num_input

np_W1 = np.random.uniform(size=(num_input, num_input_sq), low=-.1, high=.1)
np_W2 = np.random.uniform(size=(num_input_sq, num_output), low=-0.1, high=0.1)
np_b1 = np.random.uniform(size=(num_input_sq), low=-0.1, high=0.1)
np_b2 = np.random.uniform(size=(num_output), low=-0.1, high=0.1)

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
    weights = [np_W1, np_b1, np_W2, np_b2]
    data_index = batch_index * batch_size
    np_I = input_data[data_index:data_index+batch_size]
    np_T = target_data[data_index:data_index+batch_size]
    np_O, gw1, gb1, gw2, gb2, np_h2 = fn(np_I, np_W1, np_b1, np_W2, np_b2, np_T)
    grads = [gw1, gb1, gw2, gb2]
    for k in range(len(weights)):
        weights[k] -= learning_rate * grads[k]
    learning_rate *= 0.999
    count = np_T[range(batch_size),np.argmax(np_O,axis=1)].sum() + (np_T.sum(axis=1) == 0).sum()
    acc = 100.0 * count / (batch_size - (np_T.sum(axis=1) == 0).sum())
    count_list.append(acc)
    print batch_index, acc

plt.plot(range(epoch), count_list)
plt.show()
