from PIL import Image
import numpy as np
np.random.seed(0)

img = Image.open('data/2.jpg')
img = np.array(img)
img = img.reshape((480*640,3))

num_clusters = 10

def init_params():
    centers = img[np.random.choice(len(img), num_clusters, replace=False)]
    pis = np.array([1/num_clusters] * num_clusters)
    return centers, pis

centers, pis = init_params()

# def compute_new_weights_inefficient():
#     weights = np.zeros((len(img), num_clusters))
#     for i in range(len(img)):
#         if i % 10000 == 0: print(i / len(img))
#         point = img[i,:]
#         for j in range(num_clusters):
#             diff = point-centers[j]
#             num = np.exp(-1/2*(np.dot(diff.T, diff)))*pis[j]
#             weights[i,j] = num
#     # weights /= np.sum(weights, axis = 1).reshape(307200,1)
#     return weights

# def compute_new_weights():
#     weights = np.zeros((len(img), num_clusters))
#     for i in range(num_clusters):
#         diff = img - centers[i]
#         dot = np.sum(np.square(diff), axis=1)
#         weights[:, i] = np.exp(-1/2*dot) * pis[i]
#     weights /= np.sum(weights, axis=1).reshape(307200, 1)
#     return weights

def compute_new_weights():
    X = np.repeat(img[:,np.newaxis], 10, axis=1)
    X = X - centers
    weights = np.sum(np.square(X), axis=2)
    weights = np.exp(-1/2 * weights)
    weights *= pis
    weights /= np.sum(weights, axis=1).reshape(307200, 1)
    return weights


weights = compute_new_weights()
