import numpy as np


class Svm:
    def __init__(self, dims, m, n, reg):
        self.a = np.zeros(dims, dtype=float)
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

    def batch_update(self, batch, labels):
        gradient_a, gradient_b = self.gradient(batch, labels)
        eta = self.m/(self.epoch + self.n)
        self.a -= eta * gradient_a
        self.b -= eta * gradient_b

    def gradient(self, batch, labels):
        gradient_a = 0.
        gradient_b = 0.
        for (x, y) in zip(batch, labels):
            cost = max(0, 1 - y*(np.dot(self.a, x) + self.b))
            if cost == 0: continue
            gradient_a += -y * x
            gradient_b += -y
        gradient_a /= len(batch)
        gradient_b /= len(batch)
        gradient_a += self.reg * self.a
        return gradient_a, gradient_b
