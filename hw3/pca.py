import numpy as np
import numpy.linalg as la


class PCA:
    def __init__(self, data, mean, covmat):
        self.data = data
        self.evectors = None
        self.evalues = None
        self.edata = None  # the transformed data
        self.mean = mean
        self.covmat = covmat
        self.computePca()

    def computePca(self):
        evalues, evectors = la.eigh(self.covmat)  # eigh because the matrix is symmetric
        evalues = np.flip(evalues, axis=0)  # sort descending
        evectors = np.flip(evectors, axis=1)
        self.evalues = evalues
        self.evectors = evectors
        self.edata = np.dot(self.evectors.T, self.data.T)

    def reconstruct(self, num_pcs):
        copy = np.copy(self.edata)
        copy[num_pcs:, :] = 0
        recon = np.dot(self.evectors, copy) + self.mean.reshape(4, 1)
        return recon.T

    def mse(self, true_data, num_pcs):
        return np.mean(np.sum((self.reconstruct(num_pcs) - true_data) ** 2, axis=1), axis=0)
        # on piazza they said that we should follow the formula from the book: mean over
        # training examples only (not over whole matrix)
