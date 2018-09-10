import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = np.genfromtxt('data/train.csv', delimiter=',')
train = train[1:, 1:]
train_labels = train[:, 0]
train = train[:, 1:]

val = np.genfromtxt('data/val.csv', delimiter=',')
val_labels = val[1:, 0]
val = val[1:, 1:]

for num_trees in [10,30]:
    for depth in [4,16]:
        clf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth)
        clf.fit(train, train_labels)
        print('Trees: {}, depth: {}, acc: {}'.format(num_trees, depth, clf.score(val, val_labels)))

