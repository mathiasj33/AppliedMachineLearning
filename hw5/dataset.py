import numpy as np
from sklearn.cluster import KMeans
import os
import random


class Dataset:

    def __init__(self, sample_size):
        self.sample_size = sample_size
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
        # make sure that every class is represented in the test data
        test_paths.append(('data/Eat_soup/Accelerometer-2011-03-24-13-33-22-eat_soup-f1.txt',
                           self.labels_to_num['Eat_soup']))
        test_paths.append(('data/Eat_meat/Accelerometer-2011-03-24-13-12-52-eat_meat-f1.txt',
                          self.labels_to_num['Eat_meat']))
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
        overlap = True  # overlap of 0.5
        if overlap:
            for i in range(2*int(len(item) / self.sample_size)):
                index = i * self.sample_size / 2
                index = int(index)
                if len(item[index:]) < self.sample_size:
                    continue
                data.append(item[index:index + self.sample_size, :].ravel('F'))
                count += 1
        else:
            for i in range(int(len(item) / self.sample_size)):
                index = i * self.sample_size
                data.append(item[index:index + self.sample_size, :].ravel('F'))
                count += 1
            if len(item) % self.sample_size != 0:  # include last sample
                data.append(item[-self.sample_size:, :].ravel('F'))
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
        kmeans = KMeans(n_clusters=n, random_state=0, verbose=0).fit(self.train_data)
        return kmeans.cluster_centers_

    def get_file_data_with_label(self, label):
        return [x for (x, y) in self.train_data_tuple_list if y == self.labels_to_num[label]]
