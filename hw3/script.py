import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from hw3.pca import PCA


if __name__ == '__main__':
    data = np.genfromtxt('data/dataI.csv', delimiter=',', skip_header=True)
    truedat = np.genfromtxt('data/iris.csv', delimiter=',', skip_header=True)
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.cov(data, rowvar=False, bias=True)
    pca = PCA(data, mean, cov)

    mses = []
    for i in range(5):
        mses.append(pca.mse(truedat, i))

    plt.plot(mses)
    plt.show()
