import numpy as np
from PIL import Image
from sklearn.naive_bayes import BernoulliNB


def create_img(inst):
    im = Image.fromarray(inst.reshape(28,28))
    im = im.crop(im.getbbox())
    im = im.resize((20,20))
    inst[:] = 0
    inst[:400] = np.array(im).reshape(400)


def to_binary(img):
    img[img > 127] = 255.0
    img[img <= 127] = 0.0


train = np.genfromtxt('data/train.csv', delimiter=',')
train = train[1:, 1:]
train_labels = train[:, 0]
train = train[:, 1:]
to_binary(train)
np.apply_along_axis(create_img, axis=1, arr=train)
train = train[:, :400]
to_binary(train)

val = np.genfromtxt('data/val.csv', delimiter=',')
val_labels = val[1:, 0]
val = val[1:, 1:]
to_binary(val)
np.apply_along_axis(create_img, axis=1, arr=val)
val = val[:, :400]
to_binary(val)


clf = BernoulliNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))
