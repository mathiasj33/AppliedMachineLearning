import numpy as np
from sklearn.naive_bayes import GaussianNB
from PIL import Image


def to_binary(img):
    img[img > 127] = 255.0
    img[img <= 127] = 0.0


train = np.genfromtxt('data/train.csv', delimiter=',')
train = train[1:, 1:]
train_labels = train[:, 0]
train = train[:, 1:]

val = np.genfromtxt('data/val.csv', delimiter=',', max_rows=10)
val_labels = val[1:, 0]
val = val[1:, 1:]

clf = GaussianNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))

# compute mean images
test = np.genfromtxt('data/test.csv', delimiter=',')
test_preds = clf.predict(test)
for i in range(10):
    images = test[(test_preds >= i - .1) & (test_preds <= i + .1)]
    mean = np.mean(images, axis=0)
    to_binary(mean)
    mean_img = Image.fromarray(mean.reshape(28,28))
    mean_img.convert('L').save('/home/mathias/aml/means/gaussian_untouched/{}.png'.format(i))

