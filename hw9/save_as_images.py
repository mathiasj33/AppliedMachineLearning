import numpy as np
from PIL import Image

data = np.load('data/recon_data_0.2.npy')
data[data == -1] = 0
data[data == 1] = 255
for i in range(len(data)):
    Image.fromarray(data[i,:,:]).convert('L').save('data/recon_images/{}.png'.format(i))
