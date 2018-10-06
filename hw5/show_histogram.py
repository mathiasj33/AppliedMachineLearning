from hw5.dataset import Dataset
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
import matplotlib.pyplot as plt

ds = Dataset()
ds.load_and_quantize_data()
centers = np.genfromtxt('computed_data/cluster_centers.csv', delimiter=',')
nn = NN(n_neighbors=1).fit(centers)


def show_histogram(file):
    quantized, _ = ds.quantize_data_item(file)
    quantized = np.array(quantized)

    _, indices = nn.kneighbors(quantized)
    hist = np.bincount(indices.ravel(), minlength=480)
    hist = hist.astype(np.float32) / len(quantized)

    xpos = np.arange(len(hist))
    plt.figure()
    plt.bar(xpos, hist, align='center')
    plt.xticks([0, 100, 200, 300, 400, 480])
    plt.show()


for f in ds.get_file_data_with_label('Comb_hair')[:1]:
    show_histogram(f)