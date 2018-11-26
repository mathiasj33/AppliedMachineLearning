import numpy as np
np.random.seed(0)
from PIL import Image

def show(arr):
    arr[arr==-1] = 0
    arr[arr==1] = 255
    Image.fromarray(arr).show()

data = np.load('data/data.npy')

data = data.reshape(500, 28*28)
to_flip = int(0.02 * 28 * 28)
for i in range(500):
    indices = np.random.choice(np.arange(28*28), size=to_flip)
    data[i,indices] *= -1
data = data.reshape(500, 28, 28)
np.save('data/noisy_data.npy', data)