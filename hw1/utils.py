import numpy as np
from PIL import Image

MEAN_PATH = 'means'


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


def save_mean_images(clf, folder, stretched=False, max_rows=None):
    test = load_test_data(max_rows=max_rows)
    if stretched:
        np.apply_along_axis(stretched_bbox, axis=1, arr=test)
        test = test[:, :400]
    preds = clf.predict(test)
    for i in range(10):
        images = test[(preds >= i - .1) & (preds <= i + .1)]  # account for floating point inaccuracy
        mean = np.mean(images, axis=0)
        to_binary(mean)
        # mean /= 255 -- to get 0s and 1s. However, PIL then misinterprets the data.
        size = 20 if stretched else 28
        mean_img = Image.fromarray(mean.reshape(size, size))
        mean_img.convert('L').save('{}/{}/{}.png'.format(MEAN_PATH, folder, i))


def make_submission(clf, name, stretched=False, max_rows=None):
    test = load_test_data(max_rows=max_rows)
    if stretched:
        np.apply_along_axis(stretched_bbox, axis=1, arr=test)
        test = test[:, :400]
    preds = clf.predict(test)
    numbers = np.array(range(len(preds)))
    numbers_preds = np.column_stack((numbers, preds))
    np.savetxt('submission/{}.csv'.format(name), numbers_preds, fmt='%1d', delimiter=',',
               header='ImageId,Label')
