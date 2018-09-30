import numpy as np
import pickle
from PIL import Image


class Dataset:
    def __init__(self):
        self.data, self.labels, self.label_names = self.load_data()

    def load_data(self):
        data = None
        labels = []
        for i in range(1, 6):  # 6
            with open('data/cifar-10-batches-py/data_batch_{}'.format(i), 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                if i == 1:
                    data = data_dict[b'data']
                else:
                    data = np.vstack((data, data_dict[b'data']))
                labels += data_dict[b'labels']

        with open('data/cifar-10-batches-py/batches.meta', 'rb') as fo:
            label_names_dict = pickle.load(fo, encoding='bytes')
        return data, np.array(labels), [s.decode('ascii') for s in label_names_dict[b'label_names']]

    def show_img(self, data_item):
        Image.fromarray(data_item.reshape(3, 32, 32).transpose(1, 2, 0), mode='RGB').show()

    def get_label_data(self, label):
        return self.data[self.labels == label]

    def mean(self, label):
        return np.mean(self.get_label_data(label), axis=0)

    def mean_distance_matrix(self):
        dist = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                m1 = self.mean(i)
                m2 = self.mean(j)
                dist[i, j] = np.dot((m1 - m2), (m1 - m2))
                # dist[i, j] = np.linalg.norm(m1-m2)
        return dist

    @staticmethod
    def mse(d1, d2):
        return np.mean(np.sum((d1 - d2) ** 2, axis=1), axis=0)
