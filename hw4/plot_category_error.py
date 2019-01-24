from hw4.utils import Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

dataset = Dataset()
mses = []
for category in range(10):
    d = dataset.get_label_data(category)
    pca = PCA(n_components=20)
    recon = pca.fit_transform(d)  # sklearn.PCA centers automatically
    recon = pca.inverse_transform(recon)  # representing images using 20 PCs
    mses.append(Dataset.mse(d, recon))
    print("{}%".format((category+1)*10))

x_pos = np.arange(len(mses))
plt.bar(x_pos, mses, align='center')
plt.xticks(x_pos, ["{} ({})".format(i, name) for (i, name) in zip(range(1, 11), dataset.label_names)])
# on piazza it says category 1-10...
plt.ylabel('MSE (on training data)')
plt.show()
