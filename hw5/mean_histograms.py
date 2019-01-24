import numpy as np
from hw5.dataset import Dataset
import matplotlib.pyplot as plt

train_features = np.genfromtxt('computed_data/train_feature_vectors.csv', delimiter=',')
train_labels = np.genfromtxt('computed_data/train_labels.csv', delimiter=',')

features = train_features
labels = train_labels

ds = Dataset(32)
_, axes = plt.subplots(4, 4, sharex=True, sharey=True)
for i in range(4):
    for j in range(4):
        if i == 3 and j >= 2: break  # we only have 14 classes
        label = 4 * i + j
        filtered_features = features[labels==label]
        mean = np.mean(filtered_features, axis=0)
        xpos = np.arange(len(mean))
        ax = axes[i,j]
        ax.bar(xpos, mean, align='center', width=2.)
        ax.set_xticks([0, 100, 200, 300, 400, 480])
        ax.set_title('Mean histogram of {}'.format(ds.num_to_labels[label]))

plt.show()
