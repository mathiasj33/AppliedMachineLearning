from PIL import Image
import numpy as np
np.random.seed(0)
from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def em_gmm_vect(xs, pis, mus, tol=0.01, max_iter=100):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = pis[j] * mvn(mus[j]).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = ws.sum(axis=1)
        pis /= n

        mus = np.dot(ws, xs)
        mus /= ws.sum(1)[:, None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu in zip(pis, mus):
            ll_new += pi*mvn(mu).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus

img = Image.open('data/2.jpg')
img = np.array(img)
img = img.reshape((480*640,3))
img = np.asarray(img, np.float64)

num_clusters = 10

def init_params():
    centers = img[np.random.choice(len(img), num_clusters, replace=False)]
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0, verbose=1).fit(img)
    # centers = kmeans.cluster_centers_
    pis = np.array([1/num_clusters] * num_clusters)
    return centers, pis

centers, pis = init_params()

ll, pis, centers = em_gmm_vect(img, pis, centers)
for pi, c in zip(pis, centers):
    dist = np.sum(np.square(img[0] - c), axis=0)
    print(dist)
    print(np.exp(-1/2 * dist))