from hw2.utils import *
from hw2.svm import Svm
import matplotlib.pyplot as plt

train, train_labels, val, val_labels = load_my_train_val_norm_data()
epochs = 50
steps = 300
learning_rates = [(1,50)]
all_accs = []
all_weights = []

for (m,n) in learning_rates:
    svm = Svm(len(train), m, n, 1e-2)
    accuracies = []
    weights = []

    for i in range(epochs):
        held_out_indices = np.random.choice(len(train), size=50)
        held_out = train[held_out_indices, :]
        held_out_labels = train_labels[held_out_indices]
        mask = np.ones(len(train), np.bool)
        mask[held_out_indices] = False
        epoch_train = train[mask]
        epoch_train_labels = train_labels[mask]

        batch_size = int(len(train) / steps)
        batches = np.random.choice(len(epoch_train), size=(steps, batch_size), replace=False)

        for indices in batches:
            batch = epoch_train[indices]
            svm.batch_update(batch, epoch_train_labels[indices])
            if i % 30 == 0:
                accuracies.append(svm.accuracy(held_out, held_out_labels))
                weights.append(svm.magnitude())

        svm.epoch += 1

    all_accs.append(accuracies)
    all_weights.append(weights)

_, axes = plt.subplots(2, max(len(learning_rates), 2), sharex=True)
for i in range(len(learning_rates)):
    ax1 = axes[0,i]
    ax2 = axes[1,i]
    ax1.set_title('Accuracy m:{}, n:{}'.format(*learning_rates[i]))
    ax2.set_title('||w|| m:{}, n:{}'.format(*learning_rates[i]))
    ax1.plot(all_accs[i])
    ax2.plot(all_weights[i])

plt.show()
