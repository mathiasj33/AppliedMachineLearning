import numpy as np
from sklearn.cluster import KMeans
import os
import random

random.seed(0)


class Dataset:
    SEGMENT_LENGTH = 32

    def __init__(self):
        self.train_data_tuple_list = []
        self.test_data_tuple_list = []
        self.train_data = None
        self.train_labels = None
        self.labels_to_num = self.get_labels_dict()
        self.num_to_labels = {i: d for (d, i) in self.labels_to_num.items()}

    def load_and_quantize_data(self):
        self.load_data_tuple_list()
        self.quantize_data()

    def load_data_tuple_list(self):
        paths = self.get_filepaths_with_label()
        test_paths = random.sample(paths, 100)
        train_paths = [p for p in paths if p not in test_paths]
        for file, label in train_paths:
            self.train_data_tuple_list.append((np.genfromtxt(file, delimiter=' '), label))
        for file, label in test_paths:
            self.test_data_tuple_list.append((np.genfromtxt(file, delimiter=' '), label))

    def quantize_data(self):
        self.train_data = []
        self.train_labels = []
        for item, label in self.train_data_tuple_list:
            newdata, count = self.quantize_data_item(item)
            self.train_data += newdata
            self.train_labels += [label] * count
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)

    def quantize_data_item(self, item):
        data = []
        count = 0
        for i in range(int(len(item) / self.SEGMENT_LENGTH)):
            index = i * self.SEGMENT_LENGTH
            data.append(item[index:index + self.SEGMENT_LENGTH, :].ravel('F'))
            count += 1
        if len(item) % self.SEGMENT_LENGTH != 0:  # we want to use the last samples too; overlaps don't matter
            data.append(item[-self.SEGMENT_LENGTH:, :].ravel('F'))
            count += 1
        return data, count

    def get_labels_dict(self):
        return {d: i for (i, d) in enumerate([d for d in os.listdir('data') if not d.endswith('.txt')])}

    def get_filepaths_with_label(self):
        paths = [
            ('data/{}/{}'.format(dir, file), self.labels_to_num[dir])
            for dir in [d for d in os.listdir('data') if not d.endswith('.txt')]
            for file in os.listdir('data/' + dir)
        ]
        return paths

    def cluster(self, n):
        kmeans = KMeans(n_clusters=n, random_state=0, verbose=1).fit(self.train_data)
        return kmeans.cluster_centers_

    def get_file_data_with_label(self, label):
        return [x for (x, y) in self.train_data_tuple_list if y == self.labels_to_num[label]]
