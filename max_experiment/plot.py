
import matplotlib.pyplot as plt

output_files = [
        'birconv',
        'mlp100',
        'mlp10000',
        'mlp100skip',
        'mlp10000skip',
        'mlp1000-1000',
        'mlp1000-1000skip'
]

EPOCHS = 100

# Test Accuracies
#x_range = range(EPOCHS+1)
#for o in output_files:
#    accs = [0]
#    with open(o) as f:
#        for line in f.xreadlines():
#            if line.startswith('>>>'):
#                acc = float(line.split()[-1])
#                accs.append(acc)
#    plt.plot(x_range, accs, label=o)
#
#plt.legend(loc='upper left')
#plt.title('Test accuracy comparison over training')
#plt.ylabel('Accuracy')
#plt.xlabel('Iterations')
#plt.show()

# Losses
mod = 100000
x_range = range(0, 5000001, mod)
for o in output_files:
    accs = [4.5]
    with open(o) as f:
        for line in f.xreadlines():
            if line.startswith('Iter'):
                tok = line.split()
                if int(tok[1][:-1]) % mod == 0:
                    acc = float(tok[-1])
                    accs.append(acc)
    plt.plot(x_range, accs, label=o)

plt.legend(loc='lower left')
plt.title('Train loss comparison over training')
plt.ylabel('Train loss')
plt.xlabel('Iterations')
plt.show()
