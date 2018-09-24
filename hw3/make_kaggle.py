import numpy as np
from hw3.pca import PCA

data = np.genfromtxt('data/dataII.csv', delimiter=',', skip_header=True)
mean = np.mean(data, axis=0)
data -= mean
cov = np.cov(data, rowvar=False, bias=True)
pca = PCA(data, mean, cov)
recon = pca.reconstruct(2)
np.savetxt('submission/mfj3-recon.csv', recon, fmt='%.14f', delimiter=',',
           header='"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"', comments='')