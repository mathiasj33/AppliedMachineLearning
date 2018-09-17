from hw2.utils import *
from hw2.svm import Svm
import matplotlib.pyplot as plt

train, train_labels, val, val_labels = load_my_train_val_norm_data()
epochs = 50
steps = 300
regs = [1e-3, 1e-2, 1e-1, 1]
all_accs = []
all_weights = []

for reg in regs:
    svm = Svm(train.shape[1], 1, 100, reg)
    accuracies = []
    weights = []

    for i in range(epochs):
        held_out_indices = np.random.choice(len(train), size=50, replace=False)
        held_out = train[held_out_indices, :]
        held_out_labels = train_labels[held_out_indices]
        mask = np.ones(len(train), np.bool)
        mask[held_out_indices] = False
        epoch_train = train[mask]
        epoch_train_labels = train_labels[mask]

        batch_size = int(len(epoch_train) / steps)
        batches = np.random.choice(len(epoch_train), size=(steps, batch_size), replace=False)

        count = 0
        for indices in batches:
            batch = epoch_train[indices]
            svm.batch_update(batch, epoch_train_labels[indices])
            count += 1
            if count % 30 == 0:
                acc = svm.accuracy(held_out, held_out_labels)
                accuracies.append(acc)
                weights.append(svm.magnitude())

        print('Epoch {}: {}'.format(i, accuracies[-1]))
        svm.new_epoch()

    all_accs.append(accuracies)
    all_weights.append(weights)

colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
ax1 = plt.gca()
plt.figure()
ax2 = plt.gca()
for i in range(len(regs)):
    ax1.set_title('Accuracies')
    ax2.set_title(r'$||a||_2$')
    ax1.plot(all_accs[i], color=colors[i])
    ax2.plot(all_weights[i], color=colors[i])

ax1.legend([r'$\lambda={}$'.format(reg) for reg in regs])
ax2.legend([r'$\lambda={}$'.format(reg) for reg in regs])

plt.show()
