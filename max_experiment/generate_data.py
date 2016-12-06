
import numpy as np
import itertools

N_TEST = 10000
N_TRAIN = 50000

D = 100

with open('X_test.csv', 'w') as f1, open('Y_test.csv', 'w') as f2:
    for i in xrange(N_TEST):
        x = np.random.random((D)) * 4 - 2
        ind = np.argmax(x)
        y = np.zeros((D))
        y[ind] = 1
        f1.write(','.join(map(str, x)))
        f1.write('\n')
        f2.write(','.join(map(str, y)))
        f2.write('\n')

with open('X_train.csv', 'w') as f1, open('Y_train.csv', 'w') as f2:
    for i in xrange(N_TRAIN):
        x = np.random.random((D)) * 2 - 1
        ind = np.argmax(x)
        y = np.zeros((D))
        y[ind] = 1
        f1.write(','.join(map(str, x)))
        f1.write('\n')
        f2.write(','.join(map(str, y)))
        f2.write('\n')
