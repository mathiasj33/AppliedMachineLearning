import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

train_features = np.genfromtxt('computed_data/train_feature_vectors.csv', delimiter=',')
train_labels = np.genfromtxt('computed_data/train_labels.csv', delimiter=',')
test_features = np.genfromtxt('computed_data/test_feature_vectors.csv', delimiter=',')
test_labels = np.genfromtxt('computed_data/test_labels.csv', delimiter=',')
# p = np.random.permutation(len(features))
# features = features[p]
# labels = labels[p]

rfc = RFC()
rfc.fit(train_features, train_labels)
print(rfc.score(test_features, test_labels))