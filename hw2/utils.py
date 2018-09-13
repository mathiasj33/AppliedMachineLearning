import numpy as np


def label_to_numeric(label):
    if label.strip() == b'<=50K':
        return -1.
    else:
        return 1.


def load_given_train_data(max_rows=None):
    train = np.genfromtxt('data/train.data', delimiter=',', max_rows=max_rows, usecols=(0, 2, 4, 10, 11, 12, 14),
                          converters={14: label_to_numeric})
    return train


def load_my_train_val_data(max_rows=None):
    train = np.genfromtxt('data/mytrain.data', delimiter=',', max_rows=max_rows)
    train_labels = train[:, -1]
    train = train[:, :-1]
    val = np.genfromtxt('data/myval.data', delimiter=',', max_rows=max_rows)
    val_labels = val[:, -1]
    val = val[:, :-1]
    return train, train_labels, val, val_labels


def load_my_train_val_norm_data(max_rows=None):
    train = np.genfromtxt('data/mytrain_norm.data', delimiter=',', max_rows=max_rows)
    train_labels = train[:, -1]
    train = train[:, :-1]
    val = np.genfromtxt('data/myval.data', delimiter=',', max_rows=max_rows)
    val_labels = val[:, -1]
    val = val[:, :-1]

    # rescale validation data with metrics from training data
    means_stds = np.genfromtxt('data/stats.data', delimiter=',')
    val /= means_stds[1, :]
    val -= means_stds[0, :]

    return train, train_labels.astype(int), val, val_labels.astype(int)
