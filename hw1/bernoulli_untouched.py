from hw1.utils import *
from sklearn.naive_bayes import BernoulliNB

train, train_labels = load_train_data()
to_binary(train)

val, val_labels = load_val_data()
to_binary(val)

clf = BernoulliNB()
clf.fit(train, train_labels)
print(clf.score(val, val_labels))

save_mean_images(clf, 'bernoulli_untouched')
