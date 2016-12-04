
import matplotlib.pyplot as plt

output_files = [
        'birconv1.out',
        'mlp1.out'
]

EPOCHS = 100

# Test Accuracies
x_range = range(EPOCHS+1)
for o in output_files:
    accs = [0]
    with open(o) as f:
        for line in f.xreadlines():
            if line.startswith('>>>'):
                acc = float(line.split()[-1])
                accs.append(acc)
    plt.plot(x_range, accs, label=o)

plt.legend(loc='lower right')
plt.show()

# Losses
#x_range = range(0, 5000001, 100)
#for o in output_files:
#    accs = [0]
#    with open(o) as f:
#        for line in f.xreadlines():
#            if line.startswith('Iter'):
#                acc = float(line.split()[-1])
#                accs.append(acc)
#    plt.plot(x_range, accs, label=o)
#
#plt.legend(loc='lower right')
#plt.show()
