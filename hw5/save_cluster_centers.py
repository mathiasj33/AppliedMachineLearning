from hw5.dataset import Dataset
import numpy as np

ds = Dataset()
ds.load_and_quantize_data()
centers = ds.cluster(480)  # 40*12 = 480
np.savetxt('computed_data/cluster_centers.csv', centers, delimiter=',')
