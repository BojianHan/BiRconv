
import numpy as np
import itertools

N_TEST = 10000
N_TRAIN = 50000

D = 5
S = 5

selected = np.random.permutation(D)[:S]

with open('X_test.csv', 'w') as f1, open('Y_test.csv', 'w') as f2:
    for i in xrange(N_TEST):
        x = np.random.random((D)) * 2 - 1
        s = x[selected]
        ys = []
        for p in itertools.permutations(s):
            y = 1
            for j in range(S - 1):
                if p[j] > p[j+1]:
                    y = 0
                    break
            ys.append(y)
        f1.write(','.join(map(str, x)))
        f1.write('\n')
        f2.write(','.join(map(str, ys)))
        f2.write('\n')

with open('X_train.csv', 'w') as f1, open('Y_train.csv', 'w') as f2:
    for i in xrange(N_TRAIN):
        x = np.random.random((D)) * 2 - 1
        s = x[selected]
        ys = []
        for p in itertools.permutations(s):
            y = 1
            for j in xrange(S - 1):
                if p[j] > p[j+1]:
                    y = 0
                    break
            ys.append(y)
        f1.write(','.join(map(str, x)))
        f1.write('\n')
        f2.write(','.join(map(str, ys)))
        f2.write('\n')
