import random
import numpy as np

num_input = 100
num_output = 125
compare_limit = 4

selector = []
for i in range(num_output):
    selector.append(random.sample(range(num_input),compare_limit))

num_data = 100

input_data = np.random.uniform(size=(num_data, num_input), low=-1, high=1)
target_data = np.zeros((num_data, num_output))
for i in range(num_output):
    target_data[:,i] = 1 * ((np.diff(input_data[:,selector[i]]) > 0).sum(axis=1) == compare_limit - 1)

print target_data.sum(axis=1).mean()