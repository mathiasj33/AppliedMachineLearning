import numpy as np


class Svm:
    def __init__(self, dims, m, n, reg):
        self.a = np.ones(dims, dtype=float)
        self.b = 0.
        self.m = m
        self.n = n
        self.epoch = 0  # k in notes
        self.reg = reg  # lambda in notes

    def predict(self, x):
        y = np.dot(self.a, x) + self.b
        if y <= 0: return -1
        else: return 1

    def accuracy(self, data, labels):
        pred = np.apply_along_axis(self.predict, 1, data)
        return np.sum(pred == labels) / len(data)

    def magnitude(self):
        return np.sqrt(np.dot(self.a, self.a))

    def new_epoch(self):
        self.epoch += 1

    def batch_update(self, batch, labels):
        gradient_a, gradient_b = self.gradient(batch, labels)
        eta = self.m/(self.epoch + self.n)
        self.a -= eta * gradient_a
        self.b -= eta * gradient_b

    def gradient(self, batch, labels):
        gradient_a = 0.
        gradient_b = 0.
        ys = np.dot(self.a, np.transpose(batch)) + self.b  # computing everything for the batch via matrix ops
        ys *= labels
        ys = 1-ys

        it = np.nditer(ys, flags=['f_index'])  # now we have to iterate :(
        while not it.finished:
            cost = max(0, it[0])
            if cost == 0:
                it.iternext()
                continue
            i = it.index
            gradient_a += -labels[i] * batch[i, :]
            gradient_b += -labels[i]
            it.iternext()

        gradient_a /= len(batch)
        gradient_b /= len(batch)
        gradient_a += self.reg * self.a
        return gradient_a, gradient_b
