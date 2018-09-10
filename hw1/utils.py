import numpy as np
from PIL import Image


def load_train_data(max_rows=None):
    data = np.genfromtxt('data/train.csv', delimiter=',', max_rows=max_rows)
    labels = data[1:, 1]
    data = data[1:, 2:]
    return data, labels


def load_val_data(max_rows=None):
    data = np.genfromtxt('data/val.csv', delimiter=',', max_rows=max_rows)
    labels = data[1:, 0]
    data = data[1:, 1:]
    return data, labels


def load_test_data(max_rows=None):
    return np.genfromtxt('data/test.csv', delimiter=',', max_rows=max_rows)


def to_binary(data):
    data[data > 127] = 255.0
    data[data <= 127] = 0.0


def stretched_bbox(inst):
    im = Image.fromarray(inst.reshape(28, 28))
    im = im.crop(im.getbbox())
    im = im.resize((20, 20))
    inst[:] = 0
    inst[:400] = np.array(im).reshape(400)


def save_mean_images(clf, folder):
    test = load_test_data()
    preds = clf.predict(test)
    for i in range(10):
        images = test[(preds >= i - .1) & (preds <= i + .1)]  # account for floating point inaccuracy
        mean = np.mean(images, axis=0)
        to_binary(mean)
        mean /= 255
        mean_img = Image.fromarray(mean.reshape(28, 28))
        mean_img.convert('L').save('/home/mathias/aml/means/{}/{}.png'.format(folder, i))
