import numpy as np
import os


class Dataset:
    def __init__(self):
        self.data = []
        self.labels = []
        self.labels_to_num = self.get_labels_dict()
        self.num_to_labels = {i:d for (d,i) in self.labels_to_num.items()}

    def get_labels_dict(self):
        return {d:i for (i,d) in enumerate([d for d in os.listdir('data') if not d.endswith('.txt')])}

    def get_filepaths_with_label(self):
        paths = [
            ('data/{}/{}'.format(dir, file), self.labels_to_num[dir])
            for dir in [d for d in os.listdir('data') if not d.endswith('.txt')]
            for file in os.listdir('data/' + dir)
        ]
        return paths

    def load_data(self):
        for file,label in self.get_filepaths_with_label():
            self.data.append(np.genfromtxt(file, delimiter=' ').ravel('F'))
            self.labels.append(label)
