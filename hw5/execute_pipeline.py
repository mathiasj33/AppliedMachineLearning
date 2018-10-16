import random
import numpy as np

random.seed(0)
np.random.seed(0)

from hw5.save_cluster_centers import *
from hw5.save_feature_vectors import *
from hw5.classify import *

sample_size = 32
k = 480

save_cluster_centers(k, sample_size)
save_features_vectors(k, sample_size)
score, confusion = classify()
