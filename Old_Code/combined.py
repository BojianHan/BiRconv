import theano
import theano.tensor as TT
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import random

def create_network():
    # Weights
    W_io = TT.matrix('W_io')
    W_ic = TT.tensor3('W_ic')
    W_co = TT.matrix('W_co')

    # Biases
    B_c = TT.matrix('B_c')
    B_o = TT.vector('B_o')

    # Input / Target
    I = TT.matrix('input')
    T = TT.matrix('target')

    # Params of input
    batch_size = TT.shape(I)[0]
    num_input = TT.shape(I)[1]
    num_input_sq = num_input * num_input

    # Comparison
    C_row = TT.tile(I, (1, num_input)).reshape((batch_size, num_input, num_input))
    C_col = C_row.dimshuffle((0,2,1))
    new_B_c = TT.tile(B_c, (batch_size, 1, 1))
    new_W_ic = TT.tile(W_ic, (batch_size, 1, 1, 1))
    C = TT.nnet.relu(new_W_ic[:,0] * C_row + new_W_ic[:,1] * C_col + new_B_c)

    # Output
    new_B_o = TT.tile(B_o, (batch_size, 1))
    O = TT.nnet.softmax(TT.dot(I, W_io) + TT.dot(C.reshape((batch_size, num_input_sq)), W_co) + new_B_o)
    cost = TT.nnet.binary_crossentropy(O, T).mean(axis=1).sum()
    [G_io, G_ic, G_co, G_c, G_o] = TT.grad(cost, [W_io, W_ic, W_co, B_c, B_o])

    # Theano function
    fn = theano.function(
    [I, T, W_io, W_ic, W_co, B_c, B_o], 
    [O, cost, G_io, G_ic, G_co, G_c, G_o, C])
    return fn

fn = create_network()

num_input = 50
num_output = 125
compare_limit = 5
learning_rate = 1
epoch = 1000
batch_size = 500
num_data = epoch * batch_size

np_W_ic = np.random.uniform(size=(2, num_input, num_input), low=-.1, high=.1)
np_B_c = np.random.uniform(size=(num_input, num_input), low=-.1, high=.1)
np_W_io = np.random.uniform(size=(num_input, num_output), low=-.1, high=.1)
np_W_co = np.random.uniform(size=(num_input * num_input, num_output), low=-.1, high=.1)
np_B_o = np.random.uniform(size=(num_output), low=-.1, high=.1)

print "Num_Parens:", sum([np.prod(weight.shape) for weight in [np_W_io, np_W_ic, np_W_co, np_B_c, np_B_o]])

selector = []
for i in range(num_output):
    selector.append(random.sample(range(num_input),compare_limit))
input_data = np.random.uniform(size=(num_data, num_input), low=-1, high=1)
target_data = np.zeros((num_data, num_output))
for i in range(num_output):
    target_data[:,i] = 1 * ((np.diff(input_data[:,selector[i]]) > 0).sum(axis=1) == compare_limit - 1)

count_list = []
for batch_index in range(epoch):
    weights = [np_W_io, np_W_ic, np_W_co, np_B_c, np_B_o]
    data_index = batch_index * batch_size
    np_I = input_data[data_index:data_index+batch_size]
    np_T = target_data[data_index:data_index+batch_size]
    np_O, np_cost, np_G_io, np_G_ic, np_G_co, np_G_c, np_G_o, np_C = fn(np_I, np_T, np_W_io, np_W_ic, np_W_co, np_B_c, np_B_o)
    grads = [np_G_io, np_G_ic, np_G_co, np_G_c, np_G_o]
    for k in range(len(weights)):
        weights[k] -= learning_rate * grads[k]
    learning_rate *= 0.999
    count = np_T[range(batch_size),np.argmax(np_O,axis=1)].sum()
    acc = 100.0 * count / (batch_size - (np_T.sum(axis=1) == 0).sum())
    count_list.append(acc)
    print batch_index, acc
Y1 = count_list

#################################################################################

def create_mlp():
    X = TT.dmatrix('X')
    W1 = TT.dmatrix('W1')
    b1 = TT.dvector('b1')
    W2 = TT.dmatrix('W2')
    b2 = TT.dvector('b2')
    target = TT.dmatrix('target')

    bs = TT.shape(X)[0]
    h1 = TT.nnet.relu(TT.dot(X, W1) + TT.tile(b1, (bs,1)))
    output = TT.nnet.softmax(TT.dot(h1, W2) + TT.tile(b2, (bs,1)))

    cost = TT.nnet.binary_crossentropy(output, target).mean(axis=1).sum()

    [gw1, gb1, gw2, gb2] = TT.grad(cost, [W1, b1, W2, b2])
    fn = theano.function([X, W1, b1, W2, b2, target], [output, gw1, gb1, gw2, gb2, h1])
    return fn

fn = create_mlp()
num_input_sq = num_input * num_input
np_W1 = np.random.uniform(size=(num_input, num_input_sq), low=-.1, high=.1)
np_W2 = np.random.uniform(size=(num_input_sq, num_output), low=-0.1, high=0.1)
np_b1 = np.random.uniform(size=(num_input_sq), low=-0.1, high=0.1)
np_b2 = np.random.uniform(size=(num_output), low=-0.1, high=0.1)

print "Num_Parens:", sum([np.prod(weight.shape) for weight in [np_W1, np_W2, np_b1, np_b2]])

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
    count = np_T[range(batch_size),np.argmax(np_O,axis=1)].sum()
    acc = 100.0 * count / (batch_size - (np_T.sum(axis=1) == 0).sum())
    count_list.append(acc)
    print batch_index, acc
Y2 = count_list

plt.plot(range(epoch), Y1, range(epoch), Y2)
plt.show()