import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.random.seed(0)

train_features = np.genfromtxt('computed_data/train_feature_vectors.csv', delimiter=',')
train_labels = np.genfromtxt('computed_data/train_labels.csv', delimiter=',')
test_features = np.genfromtxt('computed_data/test_feature_vectors.csv', delimiter=',')
test_labels = np.genfromtxt('computed_data/test_labels.csv', delimiter=',')

rfc = RFC()
rfc.fit(train_features, train_labels)
y_pred = rfc.predict(test_features)
print(accuracy_score(test_labels, y_pred))
c = confusion_matrix(test_labels, y_pred)
class_errors = (np.sum(c, axis=1) - np.diagonal(c)) / np.sum(c, axis=1)
c = np.hstack((c, class_errors.reshape(12, 1)))
