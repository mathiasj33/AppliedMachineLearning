import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

x = tf.placeholder(tf.float32, [None, 784])
reshaped = tf.reshape(x, [-1, 28, 28, 1])
dropout_prob = tf.placeholder_with_default(1.0, shape=())

conv1 = tf.layers.conv2d(
        inputs=reshaped,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(dense, dropout_prob)
logits = tf.layers.dense(inputs=dropout, units=10)

classes = tf.argmax(input=logits, axis=1),
probabilities = tf.nn.softmax(logits)

y = tf.placeholder(tf.int64, [None])
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(classes, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('cnn_summaries_val/train')
val_writer = tf.summary.FileWriter('cnn_summaries_val/val')
test_writer = tf.summary.FileWriter('cnn_summaries_val/test')
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
val_data = mnist.validation.images
val_labels = np.asarray(mnist.validation.labels, dtype=np.int32)
test_data = mnist.test.images  # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

mini_batch_size = 100
steps = 2000
for i in range(steps):
    batch_img, batch_labels = mnist.train.next_batch(mini_batch_size)
    _, acc = sess.run((train_step, accuracy_summary), feed_dict={x: batch_img, y: batch_labels, dropout_prob: 0.4})
    train_writer.add_summary(acc, i)
    if i % 100 == 0:
        val_acc = sess.run(accuracy_summary, feed_dict={x: val_data, y:val_labels})
        val_writer.add_summary(val_acc, i)
    print(i/steps)

test_acc = sess.run(accuracy_summary, feed_dict={x: test_data, y:test_labels})
test_writer.add_summary(test_acc, steps)

train_writer.close()
val_writer.close()
test_writer.close()