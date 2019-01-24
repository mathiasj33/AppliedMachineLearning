import numpy as np

with open('data/train-images.idx3-ubyte', 'rb') as f:
    f.read(16)  # skip first 16 bytes
    data = f.read(784 * 500)  # read first 500 images

data = np.fromstring(data, dtype='uint8')
data = data.reshape(500, 28, 28)
data = np.asarray(data, dtype=np.int16)
data[data<=128] = -1
data[data>128] = 1
np.save('data/data.npy', data)