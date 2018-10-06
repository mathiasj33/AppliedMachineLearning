import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import MultinomialNB

features = np.genfromtxt('computed_data/feature_vectors.csv', delimiter=',')
labels = np.genfromtxt('computed_data/labels.csv', delimiter=',')
p = np.random.permutation(len(features))
features = features[p]
labels = labels[p]

rfc = RFC()
rfc.fit(features, labels)
print(rfc.score(features, labels))

nb = MultinomialNB()
nb.fit(features, labels)
print(nb.score(features, labels))