import numpy as np
np.random.seed(0)
from hw9.utils import show
data = np.load('data/data.npy')
recon = np.load('data/recon_data_2.npy')

num_correct = [np.sum(data[i,:,:].flatten() == recon[i,:,:].flatten()) for i in range(len(data))]
least = np.argmin(num_correct)
show(data[least])
show(recon[least])

most = np.argmax(num_correct)
show(data[most])
show(recon[most])

print('Best accuracy:')
print(np.sum(data[most].flatten() == recon[most].flatten()) / len(data[most].flatten()))

print('Worst accuracy:')
print(np.sum(data[least].flatten() == recon[least].flatten()) / len(data[least].flatten()))

print('Overall accuracy:')
print(np.sum(data.flatten() == recon.flatten()) / len(data.flatten()))

print('TPR:')
print(np.sum((data.flatten() == 1) & (recon.flatten() == 1)) / np.sum(data.flatten() == 1))

print('FPR:')
print(np.sum((data.flatten() == 1) & (recon.flatten() == -1)) / np.sum(data.flatten() == -1))