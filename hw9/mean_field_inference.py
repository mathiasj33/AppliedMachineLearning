import numpy as np
np.random.seed(0)

data = np.load('data/noisy_data.npy')

def denoise(img, theta_hh, theta_hx):
    pis = np.array([0.5] * (28 * 28)).reshape(28, 28)
    #new_pis = np.zeros((28, 28))
    pi_changes = []

    while not pi_changes or pi_changes[-1] > .01:
    #for k in range(10):
        pi_changes.append(0)
        for i in range(28):
            for j in range(28):
                left = 0
                if i > 0:
                    left += theta_hh * (2 * pis[i - 1, j] - 1)
                if i < 27:
                    left += theta_hh * (2 * pis[i + 1, j] - 1)
                if j > 0:
                    left += theta_hh * (2 * pis[i, j - 1] - 1)
                if j < 27:
                    left += theta_hh * (2 * pis[i, j + 1] - 1)
                left += theta_hx * img[i, j]

                new_pi = np.exp(left) / (np.exp(left) + np.exp(-left))
                pi_changes[-1] += np.abs(new_pi - pis[i, j])
                pis[i, j] = new_pi
        # pis = new_pis.copy()

    pis[pis > .5] = 1
    pis[pis <= .5] = -1
    return pis

def save_reconstructed(theta_hh, theta_hx):
    reconstructed = np.zeros(data.shape)
    for i in range(len(data)):
        reconstructed[i, :, :] = denoise(data[i, :, :], theta_hh, theta_hx)
        print(float(i) / len(data))

    np.save('data/recon_data_{}.npy'.format(theta_hh), reconstructed)

if __name__ == '__main__':
    theta_hx = .2
    for theta_hh in [-1, 0, .2, 1, 2]:
        save_reconstructed(theta_hh, theta_hx)