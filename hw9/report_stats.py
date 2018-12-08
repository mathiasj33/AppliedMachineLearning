import numpy as np
np.random.seed(0)
from hw9.utils import show, save
data = np.load('data/data.npy')
noisy = np.load('data/noisy_data.npy')

def print_stats(recon):
    num_correct = [np.sum(data[i, :, :].flatten() == recon[i, :, :].flatten()) for i in range(len(data))]
    least = np.argmin(num_correct)  # worst reconstruction
    most = np.argmax(num_correct)  # best reconstruction

    print('Best accuracy:')
    print(np.sum(data[most].flatten() == recon[most].flatten()) / len(data[most].flatten()))

    print('Worst accuracy:')
    print(np.sum(data[least].flatten() == recon[least].flatten()) / len(data[least].flatten()))

    print('Overall accuracy:')
    print(np.sum(data.flatten() == recon.flatten()) / len(data.flatten()))

    print('TPR:')
    print(np.sum((data.flatten() == 1) & (recon.flatten() == 1)) / np.sum(data.flatten() == 1))

    print('FPR:')
    print(np.sum((data.flatten() == -1) & (recon.flatten() == 1)) / np.sum(data.flatten() == -1))

if __name__ == '__main__':
    for i in [-1, 0, .2, 1, 2]:
        recon = np.load('data/recon_data_{}.npy'.format(i))
        print('theta_hh = {}'.format(i))
        print('===========================================')
        print_stats(recon)
        print('===========================================')
