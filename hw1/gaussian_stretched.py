from hw1.utils import *
from sklearn.naive_bayes import GaussianNB

train, train_labels = load_train_data()
np.apply_along_axis(stretched_bbox, axis=1, arr=train)
train = train[:, :400]

val, val_labels = load_val_data()
np.apply_along_axis(stretched_bbox, axis=1, arr=val)
val = val[:, :400]

clf = GaussianNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))

save_mean_images(clf, 'gaussian_stretched', stretched=True)
