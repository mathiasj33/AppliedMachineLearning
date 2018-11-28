import numpy as np
np.random.seed(0)
data = np.load('data/data.npy')
recon = np.load('data/recon_data.npy')

data = data.flatten()
recon = recon.flatten()
print(np.sum(data == recon) / len(data))