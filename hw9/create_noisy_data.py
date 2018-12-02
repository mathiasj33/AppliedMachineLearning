import numpy as np
np.random.seed(0)
from hw9.utils import show

data = np.load('data/data.npy')  # TODO: is noising working?
noisy_data = np.load('data/noisy_data.npy')
show(noisy_data[130])
show(data[130])

# data = data.reshape(500, 28*28)
# to_flip = int(0.02 * 28 * 28)
# for i in range(500):
#     indices = np.random.choice(np.arange(28*28), size=to_flip)
#     data[i,indices] *= -1
# data = data.reshape(500, 28, 28)
# np.save('data/noisy_data.npy', data)