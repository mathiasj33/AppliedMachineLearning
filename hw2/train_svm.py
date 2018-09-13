from hw2.utils import *
from hw2.svm import Svm
import matplotlib.pyplot as plt

train, train_labels, val, val_labels = load_my_train_val_norm_data()
svm = Svm(train.shape[1], 1, 50, 1e-2)
epochs = 50
steps = 300

accuracies = [svm.accuracy(val, val_labels)]

for i in range(epochs):
    for j in range(steps):
        index = np.random.choice(len(train), size=1)
        svm.batch_update([train[index].reshape(6,)], [train_labels[index]])  # batch-size of 1
    svm.epoch += 1
    accuracies.append(svm.accuracy(val, val_labels))

plt.plot(accuracies)
plt.show()
