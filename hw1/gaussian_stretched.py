import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB


def create_img(inst):
    im = Image.fromarray(inst.reshape(28,28))
    im = im.crop(im.getbbox())
    im = im.resize((20,20))
    inst[:] = 0
    inst[:400] = np.array(im).reshape(400)


train = np.genfromtxt('data/train.csv', delimiter=',')
train = train[1:, 1:]
train_labels = train[:, 0]
train = train[:, 1:]
np.apply_along_axis(create_img, axis=1, arr=train)
train = train[:, :400]

val = np.genfromtxt('data/val.csv', delimiter=',')
val_labels = val[1:, 0]
val = val[1:, 1:]
np.apply_along_axis(create_img, axis=1, arr=val)
val = val[:, :400]


clf = GaussianNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))
