import numpy as np
from sklearn.naive_bayes import BernoulliNB


def to_binary(img):
    img[img > 127] = 255.0
    img[img <= 127] = 0.0


train = np.genfromtxt('data/train.csv', delimiter=',')
train = train[1:, 1:]
train_labels = train[:, 0]
train = train[:, 1:]
to_binary(train)

val = np.genfromtxt('data/val.csv', delimiter=',')
val_labels = val[1:, 0]
val = val[1:, 1:]
to_binary(val)

clf = BernoulliNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))
