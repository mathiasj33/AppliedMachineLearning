# w=[ 0.34592437  0.06694585  0.51346522  1.14464995  0.25361073  0.30982186], b=-1.0712705563789704
from hw2.utils import *
from hw2.svm import Svm


def numeric_to_label(p):
    if p == 1:
        return '>50K'
    else:
        return '<=50K'


a = [0.34592437, 0.06694585, 0.51346522, 1.14464995, 0.25361073, 0.30982186]
b = -1.0712705563789704

test = load_test_norm_data()
means_stds = np.genfromtxt('data/stats.data', delimiter=',')
test /= means_stds[1, :]
test -= means_stds[0, :]

svm = Svm(test.shape[1], 1, 100, 1e-2)
svm.a = np.array(a)
svm.b = b
preds = list(np.apply_along_axis(svm.predict, 1, test))
preds = [numeric_to_label(p) for p in preds]
numbers = ['\'{}\''.format(i) for i in range(len(test))]
numbers_preds = np.column_stack((np.array(numbers), np.array(preds)))
np.savetxt('submission/test_sub.csv', numbers_preds, fmt='%s', delimiter=',', header='Example,Label', comments='')


