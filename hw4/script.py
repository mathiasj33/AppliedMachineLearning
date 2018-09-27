import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data():
    with open('data/cifar-10-batches-py/data_batch_1', 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    with open('data/cifar-10-batches-py/batches.meta', 'rb') as fo:
        labels_dict = pickle.load(fo, encoding='bytes')
    return data_dict, labels_dict


data_dict, labels_dict = load_data()
data = data_dict[b'data']
labels = np.array(data_dict[b'labels'])
label_names = [s.decode('ascii') for s in labels_dict[b'label_names']]


def show_img(data_item):
    Image.fromarray(data_item.reshape(3, 32, 32).transpose(1,2,0), mode='RGB').show()


def get_label_data(label):
    return data[labels == label]


def mean(label):
    return np.mean(get_label_data(label), axis=0)


def mse(d1, d2):
    return np.mean(np.sum((d1 - d2) ** 2, axis=1), axis=0)


mses = []
for category in range(10):
    d = get_label_data(category)
    pca = PCA(n_components=20)
    recon = pca.fit_transform(d)  # sklearn.PCA centers automatically
    recon = pca.inverse_transform(recon)
    mses.append(mse(d, recon))

x_pos = np.arange(len(mses))
plt.bar(x_pos, mses, align='center')
plt.xticks(x_pos, label_names)
plt.ylabel('MSE')
plt.show()