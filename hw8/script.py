from PIL import Image
import numpy as np
np.random.seed(0)
from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

img = Image.open('data/3.jpg')
width, height = img.size
img = np.array(img)
img = img.reshape((width*height,3))
img = np.asarray(img, np.float64)

num_clusters = 50
num_iter = 20

def init_params():
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, verbose=0).fit(img)
    centers = kmeans.cluster_centers_
    pis = np.array([np.sum(kmeans.labels_ == i) / len(img) for i in range(num_clusters)])
    return centers, pis

centers, pis = init_params()

def compute_new_weights():
    X = np.repeat(img[:,np.newaxis, :], num_clusters, axis=1)  # shape num_points x num_clusters x 3
    X = X - centers
    weights = np.sum(np.square(X), axis=2)  # (x-u)^T(x-u). (i,j)th entry of weights will be w_ij from book
    nn = NN(n_neighbors=1).fit(centers)
    id = nn.kneighbors(img, return_distance=False)
    closest = centers[id.reshape(width*height)]  # subtract d_min^2 for numerical stability
    d_squared = np.sum(np.square(img - closest), axis=1)
    weights -= d_squared[:, np.newaxis]
    weights = np.exp(-1/2 * weights)
    weights *= pis
    weights /= np.sum(weights, axis=1).reshape(width*height, 1)  # efficiently compute the denominator for every entry
    return weights

def compute_new_centers_and_pis(weights):
    X = np.repeat(img[:, np.newaxis, :], num_clusters, axis=1) # shape num_points x num_clusters x 3
    X *= weights[:, :, np.newaxis]  # multiply every point with corresponding cluster weights
    centers = np.sum(X, axis=0) / np.sum(weights, axis=0)[:, np.newaxis]
    pis = np.sum(weights, axis=0) / len(img)
    return centers, pis

for i in range(num_iter):
    weights = compute_new_weights()
    centers, pis = compute_new_centers_and_pis(weights)
    print(i / num_iter)

probs = np.zeros((len(img), num_clusters))
for i in range(num_clusters):
    c, pi = centers[i], pis[i]
    prob = pi * mvn(mean=c).pdf(img)
    probs[:, i] = prob

closest = np.argmax(probs, axis=1)  # obtain the cluster center with the highest posterior probability
segmented = centers[closest]
segmented = segmented.reshape((height, width, 3))
Image.fromarray(np.asarray(segmented, np.uint8)).show()