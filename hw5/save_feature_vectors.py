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


train_data = []
train_labels = []
for inst,label in ds.train_data_tuple_list:
    train_data.append(to_histogram(inst))
    train_labels.append(label)
np.savetxt('computed_data/train_feature_vectors.csv', np.array(train_data), delimiter=',')
np.savetxt('computed_data/train_labels.csv', np.array(train_labels), fmt='%d', delimiter=',')

test_data = []
test_labels = []
for inst,label in ds.test_data_tuple_list:
    test_data.append(to_histogram(inst))
    test_labels.append(label)
np.savetxt('computed_data/test_feature_vectors.csv', np.array(test_data), delimiter=',')
np.savetxt('computed_data/test_labels.csv', np.array(test_labels), fmt='%d', delimiter=',')