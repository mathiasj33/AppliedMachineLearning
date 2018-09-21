import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.genfromtxt('data/dataI.csv', delimiter=',', skip_header=True)
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.cov(data, rowvar=False, bias=True)
    eigenvalues, eigenvectors = la.eigh(cov)
    eigenvalues = np.flip(eigenvalues, axis=0)
    eigenvectors = np.flip(eigenvectors, axis=1)
    eigendata = np.dot(eigenvectors.T, data.T)

    eigendata[2:, :] = 0
    r = eigendata[:, 0]
    uc = np.dot(eigenvectors, r) + mean

    mses = [np.sum(eigenvalues[i:]) for i in range(5)]
    plt.plot(eigenvalues)
    plt.plot(mses, color='red')
    plt.show()