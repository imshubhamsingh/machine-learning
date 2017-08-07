import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def add_layer(input_data, in_size, out_size, activation_function=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(input_data, weight) + biases
    if activation_function is None:
        output = wx_plus_b
    else:
        output = activation_function(wx_plus_b)
    return output


def compute_accuracy(v_xs, v_ys):
    global prediction_layer
    y_pre = sess.run(prediction_layer, feed_dict={xs: batch_xs, ys: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction_layer = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction_layer), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(10000)
    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
    pass
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
