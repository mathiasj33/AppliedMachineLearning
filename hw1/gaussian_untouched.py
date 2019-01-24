from hw1.utils import *
from sklearn.naive_bayes import GaussianNB

train, train_labels = load_train_data()
val, val_labels = load_val_data()

clf = GaussianNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))

# save_mean_images(clf, 'gaussian_untouched')

# make_submission(clf, 'mfj3_1')
