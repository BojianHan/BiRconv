import matplotlib.pyplot as plt
import numpy as np

w = np.load('acts.npy')

#plt.matshow(w[-1,:,:])
#plt.title('Iteration: 100')
#plt.show()
for i in range(w.shape[0]):
    plt.clf()
    plt.matshow(w[i,:,:])
    plt.title('Iteration: %d' % i)
    plt.savefig('acts/img{0:02}.png' .format(i))
