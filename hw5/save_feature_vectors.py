from hw5.dataset import Dataset
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

ds = Dataset()
ds.load_data_tuple_list()
centers = np.genfromtxt('computed_data/cluster_centers.csv', delimiter=',')
nn = NN(n_neighbors=1).fit(centers)


def to_histogram(inst):
    quantized, _ = ds.quantize_data_item(inst)
    quantized = np.array(quantized)
    _, indices = nn.kneighbors(quantized)
    hist = np.bincount(indices.ravel(), minlength=480)
    hist = hist.astype(np.float32) / len(quantized)  # normalization
    return hist


data = []
labels = []
for inst,label in ds.data_tuple_list:
    data.append(to_histogram(inst))
    labels.append(label)
np.savetxt('computed_data/feature_vectors.csv', np.array(data), delimiter=',')
np.savetxt('computed_data/labels.csv', np.array(labels), fmt='%d', delimiter=',')