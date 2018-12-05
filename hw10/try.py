import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

x = tf.placeholder(tf.float32, [None, 784])
dense = tf.layers.dense(inputs=x, units=30, activation=tf.nn.sigmoid)
logits = tf.layers.dense(inputs=dense, units=10)
classes = tf.argmax(input=logits, axis=1)
probabilities = tf.nn.softmax(logits)

y = tf.placeholder(tf.int64, [None])
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(classes, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('cnn_summaries/train', sess.graph)
test_writer = tf.summary.FileWriter('cnn_summaries/test')
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

mini_batch_size = 100
steps = 200
for i in range(steps):
    batch_img, batch_labels = mnist.train.next_batch(mini_batch_size)
    _, acc = sess.run((train_step, accuracy_summary), feed_dict={x: batch_img, y: batch_labels})
    train_writer.add_summary(acc, i)
    test_acc = sess.run(accuracy_summary, feed_dict={x: eval_data, y:eval_labels})
    test_writer.add_summary(test_acc, i)
    print(i/steps)

train_writer.close()
test_writer.close()