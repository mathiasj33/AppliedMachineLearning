from hw1.utils import *
from sklearn.ensemble import RandomForestClassifier

train, train_labels = load_train_data()
val, val_labels = load_val_data()

for num_trees in [10, 30]:
    for depth in [4, 16]:
        clf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth)
        clf.fit(train, train_labels)
        print('Trees: {}, depth: {}, acc: {}'
              .format(num_trees, depth, clf.score(val, val_labels)))
