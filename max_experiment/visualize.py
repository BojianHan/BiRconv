import matplotlib.pyplot as plt
import numpy as np

w = np.load('weights.npy')

for i in range(w.shape[0]):
    plt.clf()
    plt.matshow(w[i,:,:])
    plt.savefig('weights/img{0:02}.png' .format(i))
