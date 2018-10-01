import numpy as np
from hw4.utils import Dataset
import matplotlib.pyplot as plt

dataset = Dataset()
dist = dataset.mean_distance_matrix()
one = np.ones((10, 1))
A = np.identity(10) - (1 / 10) * (np.dot(one, one.T))  # I follow the algorithm from the book
W = (-1 / 2) * np.dot(np.dot(A, dist), A.T)
Lambda, U = np.linalg.eigh(W)
Lambda = np.flip(Lambda, axis=0)  # sort descending
U = np.flip(U, axis=1)
Lambda_s = Lambda[:2]
Lambda_s = np.sqrt(Lambda_s)
U_s = U[:, :2]
Y = np.dot(U_s, np.diag(Lambda_s))

x = Y[:, 0]
y = Y[:, 1]
plt.scatter(x, y, s=100)
for i in range(len(x)):
    plt.annotate(dataset.label_names[i], (x[i], y[i]), xytext=(5,5), textcoords='offset points')
plt.show()

