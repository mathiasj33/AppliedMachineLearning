from hw5.dataset import Dataset
import numpy as np


def save_cluster_centers(k, sample_size):
    ds = Dataset(sample_size)
    ds.load_and_quantize_data()
    centers = ds.cluster(k)
    np.savetxt('computed_data/cluster_centers.csv', centers, delimiter=',')
