import theano
import theano.tensor as TT
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import random

input_layer = TT.vector('input')
target_layer = TT.vector('target')

W_io = TT.matrix('W_io')
W_ic = TT.tensor3('W_ic')
W_co = TT.matrix('W_co')
B_c = TT.matrix('B_c')
B_o = TT.vector('B_o')

pre_comp = TT.tile(input_layer, (TT.shape(input_layer)[0], 1))
comp_layer = TT.nnet.relu(W_ic[0] * pre_comp + W_ic[1] * pre_comp.T + B_c)
output_layer =  TT.nnet.softmax(TT.dot(input_layer, W_io) + TT.dot(comp_layer.flatten(), W_co) + B_o)
cost = TT.nnet.binary_crossentropy(output_layer, target_layer).mean()
#cost = ((output_layer - target_layer) ** 2).mean()

[G_io, G_ic, G_co, G_c, G_o] = TT.grad(cost, [W_io, W_ic, W_co, B_c, B_o])

fn = theano.function(
    [input_layer, target_layer, W_io, W_ic, W_co, B_c, B_o], 
    [output_layer, cost, G_io, G_ic, G_co, G_c, G_o, comp_layer])

test_fn = theano.function(
    [input_layer, W_io, W_ic, W_co, B_c, B_o], 
    [comp_layer, output_layer])

num_input = 50
num_output = 125
compare_limit = 4
learning_rate = 1
epoch = 200
batch_size = 500
num_data = epoch * batch_size

np_W_ic = np.random.uniform(size=(2, num_input, num_input), low=-.5, high=.5)
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
    update_grads = [np.zeros(weight.shape) for weight in weights]
    count = 0
    for i in range(batch_size):
        data_index = batch_index * batch_size + i
        np_I = input_data[data_index]
        np_T = target_data[data_index]
        np_O, np_cost, np_G_io, np_G_ic, np_G_co, np_G_c, np_G_o, np_C = fn(np_I, np_T, np_W_io, np_W_ic, np_W_co, np_B_c, np_B_o)
        grads = [np_G_io, np_G_ic, np_G_co, np_G_c, np_G_o]
        for j in range(len(grads)):
            update_grads[j] += grads[j]
        count = np_T[range(batch_size),np.argmax(np_O,axis=1)].sum()
    for k in range(len(weights)):
        weights[k] -= learning_rate * update_grads[k]
    print batch_index, 100.0 * count / batch_size
    count_list.append(100.0 * count / batch_size)
    learning_rate *= 0.999

plt.plot(range(epoch), count_list)
plt.show()
