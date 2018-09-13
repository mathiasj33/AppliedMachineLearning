from hw2.utils import *

train = load_given_train_data()
num_val = int(0.1*len(train))
val_indices = np.random.choice(train.shape[0], size=num_val, replace=False)
val = train[val_indices, :]
mask = np.ones(len(train), np.bool) # idea from https://stackoverflow.com/questions/25330959/how-to-select-inverse-of-indexes-of-a-numpy-array
mask[val_indices] = False
train = train[mask]

np.savetxt('data/mytrain.data', train, fmt='%1d', delimiter=',')
np.savetxt('data/myval.data', val, fmt='%1d', delimiter=',')