import numpy as np
import pickle
from PIL import Image


def load_data():
    with open('data/cifar-10-batches-py/data_batch_1', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dict = load_data()
data = dict[b'data']
labels = np.array(dict[b'labels'])


def show_img(data_item):
    Image.fromarray(data_item.reshape(3, 32, 32).transpose(1,2,0), mode='RGB').show()


def get_label_data(label):
    return data[labels == label]


def mean(label):
    return np.mean(get_label_data(label), axis=0)


show_img(mean(2))