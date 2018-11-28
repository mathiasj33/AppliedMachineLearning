import numpy as np
np.random.seed(0)
from hw9.utils import show
import matplotlib.pyplot as plt

data = np.load('data/noisy_data.npy')

theta_hh = .2
theta_hx = .2

def denoise(img):
    pis = np.array([0.5] * (28 * 28)).reshape(28, 28)
    pi_changes = []

    while not pi_changes or pi_changes[-1] > .01:
        pi_changes.append(0)
        for i in range(28):
            for j in range(28):
                left = 0
                right = 0
                if i > 0:
                    left += theta_hh * (2 * pis[i - 1, j] - 1)
                    right += -theta_hh * (2 * pis[i - 1, j] - 1)
                if i < 27:
                    left += theta_hh * (2 * pis[i + 1, j] - 1)
                    right += -theta_hh * (2 * pis[i + 1, j] - 1)
                if j > 0:
                    left += theta_hh * (2 * pis[i, j - 1] - 1)
                    right += -theta_hh * (2 * pis[i, j - 1] - 1)
                if j < 27:
                    left += theta_hh * (2 * pis[i, j + 1] - 1)
                    right += -theta_hh * (2 * pis[i, j + 1] - 1)
                left += theta_hx * img[i, j]
                right += -theta_hx * img[i, j]

                new_pi = np.exp(left) / (np.exp(left) + np.exp(right))
                pi_changes[-1] += np.abs(new_pi - pis[i, j])
                pis[i, j] = new_pi

    pis[pis > .5] = 1
    pis[pis <= .5] = -1
    return pis

reconstructed = np.zeros(data.shape)
for i in range(len(data)):
    reconstructed[i,:,:] = denoise(data[i,:,:])
    print(float(i) / len(data))

np.save('data/recon_data.npy', reconstructed)