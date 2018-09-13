from hw2.utils import *

train, train_labels, val, val_labels = load_my_train_val_data()
stds = np.std(train, axis=0)
train /= stds
means = np.mean(train, axis=0)
train -= means
# save the means because we will use them for prediction (must not use validation mean)
np.savetxt('data/stats.data', np.row_stack((means, stds)), fmt='%.10f', delimiter=',')
np.savetxt('data/mytrain_norm.data', np.column_stack((train, train_labels)), fmt='%.10f', delimiter=',')
