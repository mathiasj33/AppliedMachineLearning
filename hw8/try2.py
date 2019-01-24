from PIL import Image
import numpy as np
np.random.seed(0)
from sklearn.mixture import GaussianMixture as GM

img = Image.open('data/2.jpg')
img = np.array(img)
img = img.reshape((480*640,3))
img = np.asarray(img, np.float64)

num_clusters = 10

gmm = GM(n_components=num_clusters)
indices = gmm.fit_predict(img)

segmented = gmm.means_[indices]
segmented = segmented.reshape((480,640,3))
Image.fromarray(np.asarray(segmented, np.uint8)).show()