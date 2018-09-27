import numpy as np
from hw3.pca import PCA

true_data = np.genfromtxt('data/iris.csv', delimiter=',', skip_header=True)
true_mean = np.mean(true_data, axis=0)
true_cov = np.cov(true_data - true_mean, rowvar=False, bias=True)

output = []

for i in ['I', 'II', 'III', 'IV', 'V']:
    data = np.genfromtxt('data/data{}.csv'.format(i), delimiter=',', skip_header=True)
    mean = np.mean(data, axis=0)
    cov = np.cov(data - mean, rowvar=False, bias=True)
    true_pca = PCA(data - true_mean, true_mean, true_cov)
    pca = PCA(data - mean, mean, cov)

    mses = []
    for i in range(5):
        mses.append(true_pca.mse(true_data, i))
    for i in range(5):
        mses.append(pca.mse(true_data, i))

    output.append(mses)

output = np.array(output)
np.savetxt('submission/mfj3-numbers.csv', output, fmt='%.5f', delimiter=',',
           header='"0N","1N","2N","3N","4N","0c","1c","2c","3c","4c"', comments='')